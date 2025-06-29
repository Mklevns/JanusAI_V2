# janus/training/ppo/multi_agent_trainer.py
"""
Multi-Agent PPO Trainer with centralized training and decentralized execution.
Implements MADDPG-style centralized critic for stable multi-agent learning.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from janus.agents.multi_agent_ppo import MultiAgentPPOAgent
from janus.training.ppo.config import PPOConfig
from janus.training.ppo.multi_agent_buffer import MultiAgentRolloutBuffer
from janus.training.ppo.normalization import RunningMeanStd
from janus.training.ppo.logging_utils import setup_logging, WANDB_AVAILABLE

if WANDB_AVAILABLE:
    import wandb

logger = logging.getLogger(__name__)


class MultiAgentPPOTrainer:
    """
    Trainer for Multi-Agent PPO with centralized critic.

    Key features:
    - Centralized training: Critics observe all agents during training
    - Decentralized execution: Agents act based only on local observations
    - Shared or separate parameters for actors/critics
    - Support for cooperative, competitive, and mixed scenarios
    """

    def __init__(
        self,
        agents: List[MultiAgentPPOAgent],
        envs: List[Any],
        config: PPOConfig,
        experiment_name: str = "multi_agent_ppo",
        checkpoint_dir: Optional[Path] = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        share_critic_optimizer: bool = True,
    ):
        """
        Initialize multi-agent trainer.

        Args:
            agents: List of MultiAgentPPOAgent instances
            envs: List of environments (can be multi-agent environments)
            config: PPO configuration
            experiment_name: Name for experiment tracking
            checkpoint_dir: Directory for checkpoints
            use_tensorboard: Enable TensorBoard logging
            use_wandb: Enable Weights & Biases logging
            share_critic_optimizer: Whether to use single optimizer for all critics
        """
        self.agents = agents
        self.n_agents = len(agents)
        self.envs = envs
        self.n_envs = len(envs)
        self.config = config
        self.experiment_name = experiment_name
        self.share_critic_optimizer = share_critic_optimizer

        # Verify all agents have same configuration
        obs_dim = agents[0].observation_dim
        action_dim = agents[0].action_dim
        for agent in agents[1:]:
            assert agent.observation_dim == obs_dim, "All agents must have same obs dim"
            assert agent.action_dim == action_dim, "All agents must have same action dim"

        self.obs_shape = (obs_dim,)
        self.action_shape = () if not agents[0].continuous_actions else (action_dim,)
        self.continuous_actions = agents[0].continuous_actions

        # Device setup
        self.device = torch.device(config.device)
        for agent in self.agents:
            agent.to(self.device)

        # Create buffer
        self.buffer = MultiAgentRolloutBuffer(
            buffer_size=2048,  # Will be set properly in train()
            n_agents=self.n_agents,
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            device=self.device,
            n_envs=self.n_envs
        )

        # Reward normalization (shared across agents)
        self.reward_normalizer = RunningMeanStd() if config.normalize_rewards else None

        # Create optimizers
        self._setup_optimizers()

        # Mixed precision training
        self.scaler = (
            GradScaler()
            if config.use_mixed_precision and self.device.type == "cuda"
            else None
        )

        # Initialize environments
        self.current_obs = self._reset_all_envs()

        # Tracking
        self.global_step = 0
        self.num_updates = 0
        self.best_mean_reward = -np.inf

        # Episode tracking
        self.episode_rewards = [[] for _ in range(self.n_agents)]
        self.episode_lengths = []
        self.current_episode_rewards = np.zeros((self.n_envs, self.n_agents))
        self.current_episode_lengths = np.zeros(self.n_envs)

        # Setup logging
        self.checkpoint_dir = checkpoint_dir or Path(f"checkpoints/{experiment_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        config.to_yaml(self.checkpoint_dir / "config.yaml")

        self.loggers = setup_logging(
            experiment_name,
            self.checkpoint_dir,
            use_tensorboard,
            use_wandb,
            dict(config),
        )
        self.writer = self.loggers["writer"]

        logger.info(
            f"MultiAgentPPOTrainer initialized with {self.n_agents} agents, "
            f"{self.n_envs} environments"
        )

    def _setup_optimizers(self):
        """Setup optimizers for multi-agent training."""
        # Actor optimizers (one per agent)
        self.actor_optimizers = []
        for i, agent in enumerate(self.agents):
            optimizer = torch.optim.Adam(
                agent.actor.parameters(),
                lr=self.config.learning_rate,
                eps=1e-5
            )
            self.actor_optimizers.append(optimizer)

        # Critic optimizer(s)
        if self.share_critic_optimizer:
            # Single optimizer for all centralized critics
            critic_params = []
            for agent in self.agents:
                critic_params.extend(agent.centralized_critic.parameters())

            self.critic_optimizer = torch.optim.Adam(
                critic_params,
                lr=self.config.learning_rate,
                eps=1e-5
            )
            self.critic_optimizers = [self.critic_optimizer] * self.n_agents
        else:
            # Separate optimizer for each critic
            self.critic_optimizers = []
            for agent in self.agents:
                optimizer = torch.optim.Adam(
                    agent.centralized_critic.parameters(),
                    lr=self.config.learning_rate,
                    eps=1e-5
                )
                self.critic_optimizers.append(optimizer)

    def _reset_all_envs(self) -> np.ndarray:
        """Reset all environments and return initial observations."""
        all_obs = []

        for env in self.envs:
            obs, _ = env.reset()
            # Handle both single-agent and multi-agent environments
            if isinstance(obs, list) or (isinstance(obs, np.ndarray) and obs.ndim > 1):
                all_obs.append(obs)
            else:
                # Single agent env - replicate for all agents
                all_obs.append([obs] * self.n_agents)

        # Shape: [n_envs, n_agents, obs_dim]
        return np.array(all_obs, dtype=np.float32)

    def collect_rollouts(self, num_steps: int) -> Dict[str, float]:
        """
        Collect rollouts with decentralized execution.

        Args:
            num_steps: Number of steps to collect

        Returns:
            Dictionary of collection metrics
        """
        if self.buffer is None:
            raise RuntimeError("Buffer not initialized")

        self.buffer.clear()
        collection_start = time.time()

        for step in range(num_steps):
            # Get observations for all agents
            obs_tensor = torch.FloatTensor(self.current_obs).to(self.device)

            # Each agent selects action based on local observation
            actions = []
            log_probs = []
            values_list = []

            with torch.no_grad():
                # Decentralized action selection
                for agent_id, agent in enumerate(self.agents):
                    # Local observation for this agent
                    local_obs = obs_tensor[:, agent_id]  # [n_envs, obs_dim]

                    # Get action from local policy
                    if self.scaler and self.device.type == "cuda":
                        with autocast():
                            action, log_prob, _ = agent.act(local_obs)
                    else:
                        action, log_prob, _ = agent.act(local_obs)

                    actions.append(action)
                    log_probs.append(log_prob)

                # Get values from centralized critic
                # Prepare joint observations and actions
                joint_obs = obs_tensor.reshape(self.n_envs, -1)  # [n_envs, n_agents * obs_dim]

                # Stack actions appropriately
                if self.continuous_actions:
                    joint_actions_np = torch.stack(actions, dim=1).cpu().numpy()
                else:
                    joint_actions_np = torch.stack(actions, dim=1).cpu().numpy()

                # Get value estimates from first agent's critic (they should all be the same)
                agent_ids = torch.arange(self.n_agents).unsqueeze(0).repeat(self.n_envs, 1).to(self.device)
                all_values = self.agents[0].get_value_centralized(joint_obs, None, agent_ids)

                for agent_id in range(self.n_agents):
                    values_list.append(all_values[:, agent_id])

            # Convert to numpy for environment step
            actions_np = torch.stack(actions, dim=1).cpu().numpy()  # [n_envs, n_agents]
            log_probs_np = torch.stack(log_probs, dim=1).cpu().numpy()
            values_np = torch.stack(values_list, dim=1).cpu().numpy()

            # Step environments
            next_obs_list = []
            rewards_list = []
            dones_list = []

            for env_idx, env in enumerate(self.envs):
                # Handle different environment interfaces
                if hasattr(env, 'n_agents'):  # Multi-agent environment
                    next_obs, rewards, dones, truncated, info = env.step(actions_np[env_idx])
                else:  # Single agent environment
                    action = actions_np[env_idx, 0]  # Use first agent's action
                    next_obs, reward, done, truncated, info = env.step(action)
                    # Replicate for all agents
                    next_obs = [next_obs] * self.n_agents
                    rewards = [reward] * self.n_agents
                    dones = [done] * self.n_agents

                next_obs_list.append(next_obs)
                rewards_list.append(rewards)
                dones_list.append(dones)

                # Track episode statistics
                self.current_episode_rewards[env_idx] += rewards
                self.current_episode_lengths[env_idx] += 1

                # Check for episode end
                if any(dones) or truncated:
                    for agent_id in range(self.n_agents):
                        self.episode_rewards[agent_id].append(
                            self.current_episode_rewards[env_idx, agent_id]
                        )
                    self.episode_lengths.append(self.current_episode_lengths[env_idx])

                    # Reset tracking
                    self.current_episode_rewards[env_idx] = 0
                    self.current_episode_lengths[env_idx] = 0

                    # Reset environment
                    obs, _ = env.reset()
                    if isinstance(obs, list) or (isinstance(obs, np.ndarray) and obs.ndim > 1):
                        next_obs_list[env_idx] = obs
                    else:
                        next_obs_list[env_idx] = [obs] * self.n_agents

            # Convert to arrays
            next_obs = np.array(next_obs_list, dtype=np.float32)
            rewards = np.array(rewards_list, dtype=np.float32)
            dones = np.array(dones_list, dtype=np.float32)

            # Normalize rewards if enabled
            if self.reward_normalizer is not None:
                # Normalize across all agents
                rewards_flat = rewards.reshape(-1, 1)
                self.reward_normalizer.update(rewards_flat)
                rewards = rewards / np.sqrt(self.reward_normalizer.var + 1e-8)

            # Add to buffer
            self.buffer.add(
                obs=self.current_obs,
                action=actions_np,
                reward=rewards,
                done=dones,
                value=values_np,
                log_prob=log_probs_np,
            )

            self.current_obs = next_obs
            self.global_step += self.n_envs

        # Compute statistics
        collection_time = time.time() - collection_start

        stats = {
            "collection_time": collection_time,
            "steps_per_second": (num_steps * self.n_envs) / collection_time,
            "timesteps_collected": num_steps * self.n_envs,
        }

        # Add per-agent statistics
        for agent_id in range(self.n_agents):
            if len(self.episode_rewards[agent_id]) > 0:
                stats[f"agent_{agent_id}_mean_reward"] = np.mean(
                    self.episode_rewards[agent_id][-100:]  # Last 100 episodes
                )
                stats[f"agent_{agent_id}_std_reward"] = np.std(
                    self.episode_rewards[agent_id][-100:]
                )

        if len(self.episode_lengths) > 0:
            stats["mean_episode_length"] = np.mean(self.episode_lengths[-100:])

        return stats

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation for all agents.

        Args:
            rewards: [n_steps, n_envs, n_agents]
            values: [n_steps, n_envs, n_agents]
            dones: [n_steps, n_envs, n_agents]
            last_values: [n_envs, n_agents]

        Returns:
            advantages, returns for all agents
        """
        n_steps, n_envs, n_agents = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae_lam = torch.zeros(n_envs, n_agents, device=self.device)

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]

            delta = (
                rewards[t]
                + self.config.gamma * next_values * next_non_terminal
                - values[t]
            )
            advantages[t] = last_gae_lam = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * next_non_terminal
                * last_gae_lam
            )

        returns = advantages + values
        return advantages, returns

    def learn(
        self,
        current_clip_epsilon: float,
        current_entropy_coef: float
    ) -> Dict[str, float]:
        """
        Update all agents using centralized critics.

        Args:
            current_clip_epsilon: Current clipping parameter
            current_entropy_coef: Current entropy coefficient

        Returns:
            Dictionary of training metrics
        """
        learn_start = time.time()

        if self.buffer is None:
            raise RuntimeError("Buffer not initialized")

        # Get all data from buffer
        data = self.buffer.get()

        # Compute advantages for all agents
        with torch.no_grad():
            # Get final values for bootstrapping
            joint_obs = self.current_obs.reshape(self.n_envs, -1)
            joint_obs_tensor = torch.FloatTensor(joint_obs).to(self.device)
            agent_ids = torch.arange(self.n_agents).unsqueeze(0).repeat(self.n_envs, 1).to(self.device)

            last_values = self.agents[0].get_value_centralized(
                joint_obs_tensor, None, agent_ids
            )  # [n_envs, n_agents]

            advantages, returns = self.compute_gae(
                data["rewards"],
                data["values"],
                data["dones"],
                last_values
            )

        # Training metrics
        all_pg_losses = []
        all_value_losses = []
        all_entropy_losses = []
        all_kl_divs = []
        all_clip_fractions = []

        # Update each agent
        for agent_id, agent in enumerate(self.agents):
            # Get agent-specific data
            agent_data = self.buffer.get_agent_data(agent_id)

            # Flatten for minibatch processing
            def flatten(tensor):
                return tensor.transpose(0, 1).reshape(-1, *tensor.shape[2:])

            b_obs = flatten(agent_data["observations"])
            b_actions = flatten(agent_data["actions"]).squeeze()
            b_log_probs = flatten(agent_data["log_probs"]).squeeze()
            b_advantages = flatten(advantages[:, :, agent_id]).squeeze()
            b_returns = flatten(returns[:, :, agent_id]).squeeze()
            b_values_old = flatten(agent_data["values"]).squeeze()

            # Joint data for centralized critic
            b_joint_obs = flatten(agent_data["joint_observations"])
            b_joint_actions = flatten(agent_data["joint_actions"])
            b_agent_ids = flatten(agent_data["agent_ids"])

            # Normalize advantages
            if self.config.normalize_advantages:
                b_advantages = (b_advantages - b_advantages.mean()) / (
                    b_advantages.std() + 1e-8
                )

            # Store metrics for this agent
            pg_losses = []
            value_losses = []
            entropy_losses = []
            kl_divs = []
            clip_fractions = []

            # Train for multiple epochs
            for epoch in range(self.config.n_epochs):
                # Shuffle indices
                indices = torch.randperm(b_obs.shape[0], device=self.device)

                for start_idx in range(0, b_obs.shape[0], self.config.batch_size):
                    end_idx = start_idx + self.config.batch_size
                    batch_indices = indices[start_idx:end_idx]

                    if len(batch_indices) < self.config.batch_size // 2:
                        continue

                    try:
                        # Zero gradients
                        self.actor_optimizers[agent_id].zero_grad(set_to_none=True)
                        self.critic_optimizers[agent_id].zero_grad(set_to_none=True)

                        # Forward pass with mixed precision
                        with autocast(enabled=(self.scaler is not None and self.device.type == "cuda")):
                            # Evaluate actions with centralized critic
                            log_probs, values, entropy = agent.evaluate_with_central_critic(
                                local_observations=b_obs[batch_indices],
                                local_actions=b_actions[batch_indices].unsqueeze(-1) if b_actions[batch_indices].dim() == 1 else b_actions[batch_indices],
                                joint_observations=b_joint_obs[batch_indices],
                                joint_actions=b_joint_actions[batch_indices],
                                agent_ids=b_agent_ids[batch_indices]
                            )

                            values = values.squeeze()

                            # PPO loss computation
                            ratio = torch.exp(log_probs - b_log_probs[batch_indices])

                            # Policy loss
                            surr1 = ratio * b_advantages[batch_indices]
                            surr2 = torch.clamp(
                                ratio, 1 - current_clip_epsilon, 1 + current_clip_epsilon
                            ) * b_advantages[batch_indices]
                            policy_loss = -torch.min(surr1, surr2).mean()

                            # Value loss with clipping
                            if self.config.clip_vloss:
                                v_original = b_values_old[batch_indices]
                                v_clipped = v_original + torch.clamp(
                                    values - v_original,
                                    -current_clip_epsilon,
                                    current_clip_epsilon,
                                )
                                value_loss = 0.5 * torch.max(
                                    (values - b_returns[batch_indices]) ** 2,
                                    (v_clipped - b_returns[batch_indices]) ** 2,
                                ).mean()
                            else:
                                value_loss = 0.5 * ((values - b_returns[batch_indices]) ** 2).mean()

                            # Total loss
                            loss = (
                                policy_loss
                                + self.config.value_coef * value_loss
                                - current_entropy_coef * entropy.mean()
                            )

                        # Backward pass
                        if self.scaler and self.device.type == "cuda":
                            self.scaler.scale(loss).backward()

                            # Unscale and clip gradients
                            self.scaler.unscale_(self.actor_optimizers[agent_id])
                            self.scaler.unscale_(self.critic_optimizers[agent_id])

                            actor_grad_norm = nn.utils.clip_grad_norm_(
                                agent.actor.parameters(), self.config.max_grad_norm
                            )
                            critic_grad_norm = nn.utils.clip_grad_norm_(
                                agent.centralized_critic.parameters(), self.config.max_grad_norm
                            )

                            # Step optimizers
                            self.scaler.step(self.actor_optimizers[agent_id])
                            self.scaler.step(self.critic_optimizers[agent_id])
                            self.scaler.update()
                        else:
                            loss.backward()

                            actor_grad_norm = nn.utils.clip_grad_norm_(
                                agent.actor.parameters(), self.config.max_grad_norm
                            )
                            critic_grad_norm = nn.utils.clip_grad_norm_(
                                agent.centralized_critic.parameters(), self.config.max_grad_norm
                            )

                            self.actor_optimizers[agent_id].step()
                            self.critic_optimizers[agent_id].step()

                        # Track metrics
                        with torch.no_grad():
                            pg_losses.append(policy_loss.item())
                            value_losses.append(value_loss.item())
                            entropy_losses.append(entropy.mean().item())

                            # KL divergence
                            log_ratio = log_probs - b_log_probs[batch_indices]
                            approx_kl = (torch.exp(log_ratio) - 1 - log_ratio).mean().item()
                            kl_divs.append(approx_kl)

                            clip_fractions.append(
                                ((ratio - 1).abs() > current_clip_epsilon).float().mean().item()
                            )

                    except Exception as e:
                        logger.error(f"Training step failed for agent {agent_id}: {str(e)}", exc_info=True)
                        continue

                # Early stopping based on KL
                if kl_divs:
                    mean_kl = np.mean(kl_divs)
                    if self.config.target_kl is not None and mean_kl > self.config.target_kl:
                        logger.info(f"Early stopping for agent {agent_id} at epoch {epoch} due to KL={mean_kl:.4f}")
                        break

            # Store agent metrics
            all_pg_losses.extend(pg_losses)
            all_value_losses.extend(value_losses)
            all_entropy_losses.extend(entropy_losses)
            all_kl_divs.extend(kl_divs)
            all_clip_fractions.extend(clip_fractions)

        # Compute explained variance (using first agent's values)
        with torch.no_grad():
            agent_data = self.buffer.get_agent_data(0)
            y_pred = flatten(agent_data["values"]).squeeze()
            y_true = flatten(returns[:, :, 0]).squeeze()
            var_y = torch.var(y_true)
            explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)

        return {
            "policy_loss": np.mean(all_pg_losses) if all_pg_losses else 0,
            "value_loss": np.mean(all_value_losses) if all_value_losses else 0,
            "entropy": np.mean(all_entropy_losses) if all_entropy_losses else 0,
            "kl_divergence": np.mean(all_kl_divs) if all_kl_divs else 0,
            "clip_fraction": np.mean(all_clip_fractions) if all_clip_fractions else 0,
            "explained_variance": explained_var.item(),
            "learn_time": time.time() - learn_start,
            "actor_grad_norm": actor_grad_norm.item() if isinstance(actor_grad_norm, torch.Tensor) else actor_grad_norm,
            "critic_grad_norm": critic_grad_norm.item() if isinstance(critic_grad_norm, torch.Tensor) else critic_grad_norm,
        }

    def train(self, total_timesteps: int, rollout_length: int = 2048):
        """
        Main training loop for multi-agent PPO.

        Args:
            total_timesteps: Total environment steps to train
            rollout_length: Steps per rollout
        """
        # Reinitialize buffer with correct size
        self.buffer = MultiAgentRolloutBuffer(
            buffer_size=rollout_length,
            n_agents=self.n_agents,
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            device=self.device,
            n_envs=self.n_envs
        )

        num_updates = total_timesteps // (rollout_length * self.n_envs)
        logger.info(
            f"Starting Multi-Agent PPO training: {num_updates} updates, "
            f"{total_timesteps} total steps, {self.n_agents} agents"
        )

        start_time = time.time()

        for update in range(1, num_updates + 1):
            self.num_updates = update
            progress = update / num_updates

            # Schedule hyperparameters
            lr = self._schedule_hyperparam(
                self.config.learning_rate,
                self.config.lr_end,
                self.config.lr_schedule,
                progress
            )
            clip_eps = self._schedule_hyperparam(
                self.config.clip_epsilon,
                self.config.clip_end,
                self.config.clip_schedule,
                progress
            )
            entropy_coef = self._schedule_hyperparam(
                self.config.entropy_coef,
                self.config.entropy_end,
                self.config.entropy_schedule,
                progress
            )

            # Update learning rates
            for optimizer in self.actor_optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = lr
            for optimizer in self.critic_optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = lr

            # Collect rollouts
            rollout_metrics = self.collect_rollouts(rollout_length)

            # Learn from experience
            learn_metrics = self.learn(clip_eps, entropy_coef)

            # Combine metrics
            all_metrics = {
                **rollout_metrics,
                **learn_metrics,
                "learning_rate": lr,
                "clip_epsilon": clip_eps,
                "entropy_coef": entropy_coef,
                "update_time": time.time() - start_time,
                "total_timesteps": self.global_step,
            }

            # Log metrics
            if update % self.config.log_interval == 0:
                self._log_metrics(all_metrics, update)

            # Save checkpoint
            if update % self.config.save_interval == 0:
                self.save_checkpoint()

            # Track best performance
            mean_reward = np.mean([
                all_metrics.get(f"agent_{i}_mean_reward", -np.inf)
                for i in range(self.n_agents)
            ])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.save_checkpoint(is_best=True)

        logger.info("Multi-agent training complete!")
        self.save_checkpoint(self.checkpoint_dir / "final_model.pt")

        if self.writer:
            self.writer.close()

    def _schedule_hyperparam(
        self, initial: float, final: float, mode: str, progress: float
    ) -> float:
        """Schedule hyperparameter value based on training progress."""
        if mode == "constant":
            return initial
        elif mode == "linear":
            return initial + (final - initial) * progress
        elif mode == "cosine":
            return final + (initial - final) * 0.5 * (1 + np.cos(np.pi * progress))
        else:
            raise ValueError(f"Unknown schedule mode: {mode}")

    def _log_metrics(self, metrics: Dict[str, float], update: int) -> None:
        """Log training metrics."""
        # Console logging
        log_str = f"Update {update}: "
        for i in range(self.n_agents):
            if f"agent_{i}_mean_reward" in metrics:
                log_str += f"Agent{i}={metrics[f'agent_{i}_mean_reward']:.2f} "
        log_str += f"KL={metrics['kl_divergence']:.4f}"
        logger.info(log_str)

        # TensorBoard logging
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"train/{key}", value, self.global_step)

        # Weights & Biases logging
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(metrics, step=self.global_step)

    def save_checkpoint(
        self,
        path: Optional[Path] = None,
        is_best: bool = False
    ) -> Path:
        """Save training checkpoint."""
        if path is None:
            name = "best_model.pt" if is_best else f"checkpoint_{self.global_step}.pt"
            path = self.checkpoint_dir / name

        # Save each agent's state
        agent_states = []
        for i, agent in enumerate(self.agents):
            agent_state = {
                "actor_state_dict": agent.actor.state_dict(),
                "centralized_critic_state_dict": agent.centralized_critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizers[i].state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizers[i].state_dict(),
            }
            if agent.continuous_actions:
                agent_state["log_std"] = agent.log_std
            agent_states.append(agent_state)

        checkpoint = {
            "global_step": self.global_step,
            "num_updates": self.num_updates,
            "best_mean_reward": self.best_mean_reward,
            "config": asdict(self.config),
            "agent_states": agent_states,
            "n_agents": self.n_agents,
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        if self.reward_normalizer:
            checkpoint["reward_normalizer"] = self.reward_normalizer

        torch.save(checkpoint, path)
        logger.info(f"Multi-agent checkpoint saved to {path}")
        return path

    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Verify configuration
        assert checkpoint["n_agents"] == self.n_agents, "Number of agents mismatch"

        # Load each agent's state
        for i, (agent, agent_state) in enumerate(zip(self.agents, checkpoint["agent_states"])):
            agent.actor.load_state_dict(agent_state["actor_state_dict"])
            agent.centralized_critic.load_state_dict(agent_state["centralized_critic_state_dict"])
            self.actor_optimizers[i].load_state_dict(agent_state["actor_optimizer_state_dict"])
            self.critic_optimizers[i].load_state_dict(agent_state["critic_optimizer_state_dict"])

            if agent.continuous_actions and "log_std" in agent_state:
                agent.log_std.data = agent_state["log_std"]

        self.global_step = checkpoint["global_step"]
        self.num_updates = checkpoint["num_updates"]
        self.best_mean_reward = checkpoint.get("best_mean_reward", -np.inf)

        if "scaler_state_dict" in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "reward_normalizer" in checkpoint and self.reward_normalizer:
            self.reward_normalizer = checkpoint["reward_normalizer"]

        logger.info(f"Loaded multi-agent checkpoint from {path}")
