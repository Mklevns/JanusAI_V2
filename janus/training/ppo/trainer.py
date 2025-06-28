# File: janus/training/ppo/trainer.py
"""
PPO Trainer for Proximal Policy Optimization (PPO) in JanusAI V2.

This module implements a PPO trainer that manages the training loop,
handles rollouts, normalizes rewards, and integrates w/
TensorBoard and Weights & Biases.

Features:
    - Asynchronous rollout collection with multiple environments
    - Mixed precision training support
    - Comprehensive hyperparameter scheduling
    - Robust error handling and recovery
    - Advanced logging and metrics tracking
    - Checkpointing with best model tracking
    - Evaluation during training
    - Memory-efficient buffer management
"""

# Fixed and optimized PPO Trainer for JanusAI V2
# Includes reproducibility seeding, cleaned imports, best practices

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from janus.rewards import RewardHandler
from .buffer import RolloutBuffer
from .collector import AsyncRolloutCollector
from .config import PPOConfig
from .logging_utils import setup_logging, WANDB_AVAILABLE
from .normalization import RunningMeanStd

if WANDB_AVAILABLE:
    import wandb

logger = logging.getLogger(__name__)

# Set reproducibility seeds
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PPOTrainer:
    """
    Production-ready PPO trainer with advanced features.

    This trainer orchestrates the entire PPO training process including:
    - Asynchronous environment stepping
    - Advantage estimation with GAE
    - Policy and value network updates
    - Hyperparameter scheduling
    - Comprehensive logging and checkpointing
    - Mixed precision training (AMP) and gradient accumulation
    - Safe GPU memory management and checkpointing
    """

    def __init__(
        self,
        agent,
        envs: List,
        config: PPOConfig,
        experiment_name: str = "ppo_experiment",
        checkpoint_dir: Optional[Path] = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        eval_env: Optional[Any] = None,
    ):
        self.agent = agent
        self.envs = envs
        self.config = config
        self.eval_env = eval_env
        self.experiment_name = experiment_name

        self.device = torch.device(config.device)
        self.agent.to(self.device)
        logger.info("Using device: %s", self.device)

        self.n_envs = len(envs)
        dummy_obs, _ = envs[0].reset()
        self.obs_shape = dummy_obs.shape

        if hasattr(envs[0].action_space, "n"):
            self.action_shape = ()
            self.action_dim = envs[0].action_space.n
            self.continuous_actions = False
        else:
            self.action_shape = envs[0].action_space.shape
            self.action_dim = envs[0].action_space.shape[0]
            self.continuous_actions = True

        logger.info(
            "Environment: obs_shape=%s, action_dim=%s, continuous=%s",
            self.obs_shape,
            self.action_dim,
            self.continuous_actions,
        )

        self.collector = AsyncRolloutCollector(
            envs, self.device, config.num_workers, config.use_subprocess_envs
        )
        self.executor = (
            ThreadPoolExecutor(max_workers=config.num_workers)
            if config.num_workers > 1
            else None
        )
        self.buffer = None
        self._current_values = None

        self.reward_normalizer = RunningMeanStd() if config.normalize_rewards else None

        self.optimizer = torch.optim.Adam(
            agent.parameters(), lr=config.learning_rate, eps=1e-5
        )
        self.scaler = (
            GradScaler()
            if config.use_mixed_precision and self.device.type == "cuda"
            else None
        )

        self.global_step = 0
        self.num_updates = 0
        self.best_mean_reward = -np.inf
        self.best_eval_reward = -np.inf

        self.reward_handler = None  # Placeholder for future reward handler integration
        self.debug_rewards = getattr(config, "debug_rewards", False)

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
        self.csv_logger = self.loggers["csv"]

        logger.info(
            "PPOTrainer initialized with %d environments, %d workers",
            self.n_envs,
            config.num_workers,
        )

    def evaluate(
        self,
        eval_env: Optional[Any] = None,
        num_episodes: int = 10,
        render: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate the agent on the evaluation environment.

        Args:
            eval_env: Environment to evaluate on (uses self.eval_env if None)
            num_episodes: Number of episodes to run
            render: Whether to render the environment

        Returns:
            Dictionary of evaluation metrics
        """
        env = eval_env or self.eval_env
        if env is None:
            logger.warning("No evaluation environment provided")
            return {"eval_reward": 0.0}

        self.agent.eval()  # Set model to eval mode for inference only

        episode_rewards = np.zeros(num_episodes, dtype=np.float32)
        episode_lengths = np.zeros(num_episodes, dtype=np.int32)

        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            reward_sum = 0
            step_count = 0

            while not done:
                if render:
                    env.render()

                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(
                    0
                )  # device-safe tensor conversion

                with torch.no_grad():
                    action, _ = self.agent.act(obs_tensor, deterministic=True)

                if self.continuous_actions:
                    action_np = action.cpu().numpy()[0]
                else:
                    action_np = action.cpu().numpy()[0].item()

                obs, reward, done, truncated, _ = env.step(action_np)
                reward_sum += reward
                step_count += 1

                if truncated:
                    done = True

            episode_rewards[ep] = reward_sum
            episode_lengths[ep] = step_count

        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear unused GPU memory post-eval

        return {
            "eval_reward": episode_rewards.mean(),
            "eval_reward_std": episode_rewards.std(),
            "eval_length": episode_lengths.mean(),
        }

    def collect_rollouts(self, num_steps: int) -> Dict[str, float]:
        """
        Collect rollouts from environments with modular reward computation.
        This version properly integrates with AsyncRolloutCollector.collect()
        and supports the RewardHandler for modular rewards.
        """
        self.buffer.clear()
        collection_start = time.time()

        # Initialize reward tracking for this rollout
        component_rewards_sum = {}
        component_rewards_count = 0

        for step in range(num_steps):
            # Create action getter function that collector will call
            def get_actions_fn(obs_tensor):
                """Get actions and log_probs for given observations."""
                with torch.no_grad():
                    if self.scaler:
                        with autocast():
                            actions, log_probs = self.agent.act(obs_tensor)
                    else:
                        actions, log_probs = self.agent.act(obs_tensor)

                return actions.cpu().numpy(), log_probs.cpu().numpy()

            # Collect step - now returns 7 values including infos
            (
                original_obs,
                next_obs,
                env_rewards,
                dones,
                truncateds,
                log_probs_np,
                infos,
            ) = self.collector.collect(get_actions_fn, self.executor)

            # Get values for the original observations
            obs_tensor = torch.FloatTensor(original_obs).to(self.device)
            with torch.no_grad():
                if self.scaler:
                    with autocast():
                        # Re-compute actions to ensure consistency
                        actions, _ = self.agent.act(obs_tensor)
                        values = self.agent.critic(obs_tensor).squeeze(-1)
                else:
                    actions, _ = self.agent.act(obs_tensor)
                    values = self.agent.critic(obs_tensor).squeeze(-1)

            actions_np = actions.cpu().numpy()
            values_np = values.cpu().numpy()

            # Compute rewards using RewardHandler if available
            if self.reward_handler is not None:
                modular_rewards = np.zeros(self.n_envs, dtype=np.float32)

                for env_idx in range(self.n_envs):
                    # Compute modular reward for each environment
                    reward = self.reward_handler.compute_reward(
                        observation=original_obs[env_idx],
                        action=actions_np[env_idx],
                        next_observation=next_obs[env_idx],
                        info=infos[env_idx],
                    )
                    modular_rewards[env_idx] = reward

                    # Track component breakdown if debugging
                    if self.debug_rewards and step % 100 == 0:  # Log every 100 steps
                        breakdown = self.reward_handler.get_component_breakdown(
                            original_obs[env_idx],
                            actions_np[env_idx],
                            next_obs[env_idx],
                            infos[env_idx],
                        )
                        for comp_name, comp_reward in breakdown.items():
                            if comp_name not in component_rewards_sum:
                                component_rewards_sum[comp_name] = 0.0
                            component_rewards_sum[comp_name] += comp_reward
                        component_rewards_count += 1

                # Use modular rewards instead of environment rewards
                rewards = modular_rewards
            else:
                # Use raw environment rewards
                rewards = env_rewards

            # Apply reward normalization if configured
            if self.reward_normalizer is not None:
                self.reward_normalizer.update(rewards.reshape(-1, 1))
                normalized_rewards = rewards / np.sqrt(
                    self.reward_normalizer.var + 1e-8
                )
            else:
                normalized_rewards = rewards

            # Store in buffer
            self.buffer.add(
                obs=original_obs,
                action=actions_np,
                reward=normalized_rewards,
                done=dones,
                value=values_np,
                log_prob=log_probs_np,
            )

            # Handle episode resets for reward handler
            if self.reward_handler is not None:
                for env_idx in range(self.n_envs):
                    if dones[env_idx] or truncateds[env_idx]:
                        # Reset reward components for completed episodes
                        # Note: This resets all components - you might want per-env handlers
                        self.reward_handler.reset()
                        break  # Only reset once per step

            self.global_step += self.n_envs

        collection_time = time.time() - collection_start

        # Get statistics
        stats = self.collector.get_statistics()
        stats["collection_time"] = collection_time
        stats["steps_per_second"] = (num_steps * self.n_envs) / collection_time

        # Add reward component statistics if available
        if self.debug_rewards and component_rewards_count > 0:
            for comp_name, total in component_rewards_sum.items():
                avg_reward = total / component_rewards_count
                stats[f"reward_component/{comp_name}"] = avg_reward

        return stats

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Rewards tensor [n_steps, n_envs]
            values: Value estimates [n_steps, n_envs]
            dones: Done flags [n_steps, n_envs]
            last_values: Bootstrap values [n_envs]

        Returns:
            Tuple of (advantages, returns)
        """
        n_steps, n_envs = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae_lam = torch.zeros(n_envs, device=self.device)

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
        self, current_clip_epsilon: float, current_entropy_coef: float
    ) -> Dict[str, float]:
        """
        Perform PPO update from collected rollouts.

        Args:
            current_clip_epsilon: Current clip ratio for PPO
            current_entropy_coef: Current entropy bonus coefficient

        Returns:
            Training metrics
        """
        learn_start = time.time()
        data = self.buffer.get()

        with torch.no_grad():
            obs_tensor = torch.tensor(
                self.collector.current_obs, dtype=torch.float32, device=self.device
            )
            last_values = self.agent.critic(obs_tensor).squeeze(-1)

            advantages, returns = self.compute_gae(
                data["rewards"], data["values"], data["dones"], last_values
            )

        def flatten(tensor):
            return tensor.swapaxes(0, 1).reshape(-1, *tensor.shape[2:])

        b_obs = flatten(data["observations"])
        b_actions = flatten(data["actions"])
        b_log_probs = flatten(data["log_probs"]).squeeze()
        b_advantages = flatten(advantages).squeeze()
        b_returns = flatten(returns).squeeze()

        if self.config.normalize_advantages:
            b_advantages = (b_advantages - b_advantages.mean()) / (
                b_advantages.std() + 1e-8
            )

        pg_losses, value_losses, entropy_losses, kl_divs, clip_fractions = (
            [],
            [],
            [],
            [],
            [],
        )
        total_batch_size = (
            self.config.batch_size * self.config.gradient_accumulation_steps
        )

        for epoch in range(self.config.n_epochs):
            indices = np.random.permutation(b_obs.shape[0])
            accumulation_counter = 0

            for start_idx in range(0, b_obs.shape[0], self.config.batch_size):
                batch_indices = indices[start_idx : start_idx + self.config.batch_size]
                if len(batch_indices) < self.config.batch_size // 2:
                    continue

                try:
                    if self.scaler:
                        with autocast():
                            log_probs, values, entropy = self.agent.evaluate(
                                b_obs[batch_indices], b_actions[batch_indices]
                            )
                    else:
                        log_probs, values, entropy = self.agent.evaluate(
                            b_obs[batch_indices], b_actions[batch_indices]
                        )

                    values = values.squeeze()
                    ratio = torch.exp(log_probs - b_log_probs[batch_indices])
                    surr1 = ratio * b_advantages[batch_indices]
                    surr2 = (
                        torch.clamp(
                            ratio, 1 - current_clip_epsilon, 1 + current_clip_epsilon
                        )
                        * b_advantages[batch_indices]
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_pred_clipped = data["values"].flatten()[
                        batch_indices
                    ] + torch.clamp(
                        values - data["values"].flatten()[batch_indices],
                        -current_clip_epsilon,
                        current_clip_epsilon,
                    )
                    value_loss = (
                        0.5
                        * torch.max(
                            (value_pred_clipped - b_returns[batch_indices]) ** 2,
                            (values - b_returns[batch_indices]) ** 2,
                        ).mean()
                    )

                    loss = (
                        policy_loss
                        + self.config.value_coef * value_loss
                        - current_entropy_coef * entropy.mean()
                    )
                    loss /= self.config.gradient_accumulation_steps

                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    accumulation_counter += 1

                    if (
                        accumulation_counter % self.config.gradient_accumulation_steps
                        == 0
                    ):
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(
                                self.agent.parameters(), self.config.max_grad_norm
                            )
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            nn.utils.clip_grad_norm_(
                                self.agent.parameters(), self.config.max_grad_norm
                            )
                            self.optimizer.step()

                        self.optimizer.zero_grad()

                    with torch.no_grad():
                        pg_losses.append(
                            policy_loss.item() * self.config.gradient_accumulation_steps
                        )
                        value_losses.append(value_loss.item())
                        entropy_losses.append(entropy.mean().item())
                        kl_divs.append(
                            (b_log_probs[batch_indices] - log_probs).mean().item()
                        )
                        clip_fractions.append(
                            (torch.abs(ratio - 1) > current_clip_epsilon)
                            .float()
                            .mean()
                            .item()
                        )

                except Exception as e:
                    logger.error(f"Training error at epoch {epoch}: {e}", exc_info=True)
                    continue

            mean_kl = np.mean(kl_divs) if kl_divs else 0
            if self.config.target_kl and mean_kl > self.config.target_kl:
                logger.info(f"Early stopping at epoch {epoch} due to KL={mean_kl:.4f}")
                break

        with torch.no_grad():
            y_pred = data["values"].flatten()
            y_true = returns.flatten()
            explained_var = 1 - torch.var(y_true - y_pred) / (torch.var(y_true) + 1e-8)

            return {
                "policy_loss": np.mean(pg_losses),
                "value_loss": np.mean(value_losses),
                "entropy": np.mean(entropy_losses),
                "kl_divergence": np.mean(kl_divs),
                "clip_fraction": np.mean(clip_fractions),
                "explained_variance": explained_var.item(),
                "learn_time": time.time() - learn_start,
            }

        def train(self, total_timesteps: int, rollout_length: int = 2048):
            """
            Main PPO training loop.

            Args:
                total_timesteps: Total steps to train on
                rollout_length: Number of env steps per rollout
            """
            self.buffer = RolloutBuffer(
                buffer_size=rollout_length,
                obs_shape=self.obs_shape,
                action_shape=self.action_shape,
                device=self.device,
                n_envs=self.n_envs,
            )

            num_updates = total_timesteps // (rollout_length * self.n_envs)
            logger.info(
                f"Starting PPO training: {num_updates} updates, {total_timesteps} steps"
            )
            start_time = time.time()

            for update in range(1, num_updates + 1):
                self.num_updates = update
                progress = update / num_updates

                lr = self._schedule_hyperparam(
                    self.config.learning_rate,
                    self.config.lr_end,
                    self.config.lr_schedule,
                    progress,
                )
                clip_eps = self._schedule_hyperparam(
                    self.config.clip_epsilon,
                    self.config.clip_end,
                    self.config.clip_schedule,
                    progress,
                )
                entropy_coef = self._schedule_hyperparam(
                    self.config.entropy_coef,
                    self.config.entropy_end,
                    self.config.entropy_schedule,
                    progress,
                )

                for group in self.optimizer.param_groups:
                    group["lr"] = lr

                rollout_metrics = self.collect_rollouts(rollout_length)
                learn_metrics = self.learn(clip_eps, entropy_coef)

                all_metrics = {
                    **rollout_metrics,
                    **learn_metrics,
                    "learning_rate": lr,
                    "clip_epsilon": clip_eps,
                    "entropy_coef": entropy_coef,
                    "update_time": time.time() - start_time,
                }

                if update % self.config.log_interval == 0:
                    logger.info(
                        f"Update {update}: Reward {all_metrics.get('mean_episode_reward', 0):.2f} | KL {all_metrics['kl_divergence']:.4f}"
                    )
                    self._log_metrics(all_metrics)

                if update % self.config.save_interval == 0:
                    self.save_checkpoint()

                if self.eval_env and update % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate(num_episodes=10)
                    logger.info(
                        f"Eval @ Update {update}: Reward {eval_metrics['eval_reward']:.2f} ± {eval_metrics['eval_reward_std']:.2f}"
                    )
                    self._log_metrics(eval_metrics, prefix="eval")

                    if eval_metrics["eval_reward"] > self.best_eval_reward:
                        self.best_eval_reward = eval_metrics["eval_reward"]
                        self.save_checkpoint(is_best=True)

                if (
                    all_metrics.get("mean_episode_reward", -np.inf)
                    > self.best_mean_reward
                ):
                    self.best_mean_reward = all_metrics["mean_episode_reward"]
                    if not self.eval_env:
                        self.save_checkpoint(is_best=True)

            logger.info("Training complete")
            self.save_checkpoint(self.checkpoint_dir / "final_model.pt")
            if self.writer:
                self.writer.close()
            if self.executor:
                self.executor.shutdown(wait=True)

    def _schedule_hyperparam(
        self, initial: float, final: float, mode: str, progress: float
    ) -> float:
        if mode == "constant":
            return initial
        elif mode == "linear":
            return initial + (final - initial) * progress
        elif mode == "cosine":
            return final + (initial - final) * 0.5 * (1 + np.cos(np.pi * progress))
        else:
            raise ValueError(f"Unsupported schedule type: {mode}")

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "train"):
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, self.global_step)
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(
                {f"{prefix}/{k}": v for k, v in metrics.items()}, step=self.global_step
            )

    def save_checkpoint(
        self, path: Optional[Path] = None, is_best: bool = False
    ) -> Path:
        if path is None:
            name = "best_model.pt" if is_best else f"checkpoint_{self.global_step}.pt"
            path = self.checkpoint_dir / name

        checkpoint = {
            "global_step": self.global_step,
            "num_updates": self.num_updates,
            "best_mean_reward": self.best_mean_reward,
            "best_eval_reward": self.best_eval_reward,
            "agent_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
        }
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        if self.reward_normalizer:
            checkpoint["reward_normalizer"] = self.reward_normalizer

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
        return path

    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.num_updates = checkpoint["num_updates"]
        self.best_mean_reward = checkpoint.get("best_mean_reward", -np.inf)
        self.best_eval_reward = checkpoint.get("best_eval_reward", -np.inf)
        if "scaler_state_dict" in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "reward_normalizer" in checkpoint and self.reward_normalizer:
            self.reward_normalizer = checkpoint["reward_normalizer"]
        logger.info(f"Loaded checkpoint from {path}")
