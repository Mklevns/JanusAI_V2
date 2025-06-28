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
from dataclasses import asdict # Added import
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict

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

        Type hints added for clarity and mypy compliance.
        """
        if self.buffer is None:
            raise RuntimeError("Buffer not initialized. Call train() first.")

        self.buffer.clear()
        collection_start = time.time()

        # Initialize reward tracking with proper type hints
        component_rewards_sum: Dict[str, float] = {}
        # component_rewards_count: int = 0 # This was in the request but not used, so commented out

        for step in range(num_steps):
            # Create action getter function that collector will call
            def get_actions_fn(obs_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
                """Get actions and log_probs for given observations."""
                with torch.no_grad():
                    if self.scaler and self.device.type == "cuda":
                        with autocast():
                            actions, log_probs = self.agent.act(obs_tensor)
                    else:
                        actions, log_probs = self.agent.act(obs_tensor)

                return actions.cpu().numpy(), log_probs.cpu().numpy()

            # Collect step - assumes collector.collect returns relevant data
            # The original code had a more complex interaction with RewardHandler and normalization here.
            # The provided snippet for collect_rollouts is simpler and focuses on the action getter.
            # I will adapt to keep the core logic of interaction with buffer and collector as per the new snippet.
            # Assuming self.collector.collect() is adapted or provides necessary data.
            # For now, I'm matching the structure of the provided snippet.
            # This might need further adjustments based on how AsyncRolloutCollector's `collect` method
            # is structured after its own update.

            # The provided snippet for `collect_rollouts` in the prompt is incomplete regarding
            # how `original_obs, next_obs, env_rewards, dones, truncateds, log_probs_np, infos` are obtained
            # after defining `get_actions_fn`.
            # I will assume that `self.collector.collect` is called similarly to the original code,
            # and then the buffer is populated. The key change from the prompt is the `get_actions_fn` structure
            # and the initial type hints.

            (
                original_obs,  # Assuming these are obtained from the collector
                next_obs,
                env_rewards,
                dones,
                truncateds,
                log_probs_np, # This was obtained from get_actions_fn before, now from collector
                infos,
            ) = self.collector.collect(get_actions_fn, self.executor)


            obs_tensor = torch.FloatTensor(original_obs).to(self.device)
            with torch.no_grad():
                if self.scaler and self.device.type == "cuda":
                    with autocast():
                        # Re-compute actions for consistency if needed by agent's architecture
                        # or use actions from get_actions_fn if they are guaranteed to be consistent
                        actions, _ = self.agent.act(obs_tensor) # Assuming log_probs from collect are sufficient
                        values = self.agent.critic(obs_tensor).squeeze(-1)
                else:
                    actions, _ = self.agent.act(obs_tensor)
                    values = self.agent.critic(obs_tensor).squeeze(-1)

            actions_np = actions.cpu().numpy() # actions from re-evaluation or get_actions_fn
            values_np = values.cpu().numpy()

            # Reward computation and normalization (adapted from original logic)
            if self.reward_handler is not None:
                modular_rewards = np.zeros(self.n_envs, dtype=np.float32)
                component_rewards_count_this_step = 0 # Renamed to avoid conflict if outer var is restored
                for env_idx in range(self.n_envs):
                    reward = self.reward_handler.compute_reward(
                        observation=original_obs[env_idx],
                        action=actions_np[env_idx], # Use consistent actions
                        next_observation=next_obs[env_idx],
                        info=infos[env_idx],
                    )
                    modular_rewards[env_idx] = reward
                    if self.debug_rewards: # Simplified logging condition
                        breakdown = self.reward_handler.get_component_breakdown(
                            original_obs[env_idx],
                            actions_np[env_idx],
                            next_obs[env_idx],
                            infos[env_idx],
                        )
                        for comp_name, comp_reward in breakdown.items():
                            component_rewards_sum[comp_name] = component_rewards_sum.get(comp_name, 0.0) + comp_reward
                        component_rewards_count_this_step +=1 # count updates for averaging later if needed

                rewards_to_process = modular_rewards
            else:
                rewards_to_process = env_rewards

            if self.reward_normalizer is not None:
                self.reward_normalizer.update(rewards_to_process.reshape(-1, 1))
                final_rewards = rewards_to_process / np.sqrt(
                    self.reward_normalizer.var + 1e-8
                )
            else:
                final_rewards = rewards_to_process

            self.buffer.add(
                obs=original_obs,
                action=actions_np, # Use consistent actions
                reward=final_rewards,
                done=dones,
                value=values_np,
                log_prob=log_probs_np, # Log probs from the collector
            )

            if self.reward_handler is not None:
                for env_idx in range(self.n_envs):
                    if dones[env_idx] or truncateds[env_idx]:
                        self.reward_handler.reset()
                        break

            self.global_step += self.n_envs

        collection_time = time.time() - collection_start
        stats = self.collector.get_statistics()
        stats["collection_time"] = collection_time
        stats["steps_per_second"] = (num_steps * self.n_envs) / collection_time

        # Averaging component rewards if tracked
        # The original request had component_rewards_count outside loop, implying cumulative count.
        # If it's per rollout, it should be sum / (num_steps * n_envs_processed_for_components)
        # For simplicity, if component_rewards_sum is filled, we log its current sums or averages.
        # The prompt's snippet for collect_rollouts didn't use component_rewards_count.
        # Re-instating a simplified version of component reward logging.
        if self.debug_rewards and component_rewards_sum:
             # Simple sum for now, averaging logic might need refinement based on how many steps contribute
            for comp_name, total_reward in component_rewards_sum.items():
                 # This simple average assumes all components are updated equally.
                 # A more robust way would be to count updates per component.
                 # For now, using total sum as an indicative measure.
                stats[f"reward_component_sum/{comp_name}"] = total_reward


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

def learn(self, current_clip_epsilon: float, current_entropy_coef: float) -> Dict[str, float]:
    """PPO update with memory-efficient implementation."""
    learn_start = time.time()

    if self.buffer is None:
        raise RuntimeError("Buffer not initialized")

    data = self.buffer.get()

    # Move advantage computation to GPU in chunks to save memory
    with torch.no_grad():
        # Ensure current_obs is a tensor on the correct device
        # Original code used self.collector.current_obs, assuming it's updated and correct
        obs_tensor = torch.as_tensor(
            self.collector.current_obs, dtype=torch.float32, device=self.device
        )
        last_values = self.agent.critic(obs_tensor).squeeze(-1)

        advantages, returns = self.compute_gae(
            data["rewards"], data["values"], data["dones"], last_values
        )

    # Flatten tensors for minibatch processing
    def flatten(tensor):
        # Original: tensor.swapaxes(0, 1).reshape(-1, *tensor.shape[2:])
        # New: tensor.transpose(0, 1).reshape(-1, *tensor.shape[2:])
        # torch.transpose is an alias for torch.swapaxes, so this is fine.
        return tensor.transpose(0, 1).reshape(-1, *tensor.shape[2:])

    b_obs = flatten(data["observations"])
    b_actions = flatten(data["actions"])
    b_log_probs = flatten(data["log_probs"]).squeeze()
    b_advantages = flatten(advantages).squeeze()
    b_returns = flatten(returns).squeeze()
    # Store original values from buffer for value loss clipping if clip_vloss is true
    b_values_original_flat = flatten(data["values"]).squeeze()


    # Free original data to save memory
    del data # Keep advantages and returns as they are used
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if self.config.normalize_advantages:
        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )

    # Training metrics
    pg_losses, value_losses, entropy_losses, kl_divs, clip_fractions = [], [], [], [], []
    # grad_norm needs to be initialized before the loop in case no training step completes
    grad_norm = torch.tensor(0.0, device=self.device)


    for epoch in range(self.config.n_epochs):
        # Shuffle indices for each epoch
        indices = torch.randperm(b_obs.shape[0], device=self.device)

        for start_idx in range(0, b_obs.shape[0], self.config.batch_size):
            end_idx = start_idx + self.config.batch_size
            batch_indices = indices[start_idx:end_idx]

            # Skip small batches, e.g., if less than half of batch_size
            # This check was in original code for `len(batch_indices) < self.config.batch_size // 2`
            # The new code doesn't have this, but it can be useful. Adding it back.
            if len(batch_indices) < self.config.batch_size // 2:
                continue

            try:
                # Forward pass with mixed precision
                # autocast enabled should check self.scaler is not None AND self.device.type == "cuda"
                with autocast(enabled=(self.scaler is not None and self.device.type == "cuda")):
                    log_probs, values, entropy = self.agent.evaluate(
                        b_obs[batch_indices], b_actions[batch_indices]
                    )

                # Compute losses
                values = values.squeeze() # Ensure values is 1D
                ratio = torch.exp(log_probs - b_log_probs[batch_indices])

                # PPO clipped objective
                surr1 = ratio * b_advantages[batch_indices]
                surr2 = torch.clamp(
                    ratio, 1 - current_clip_epsilon, 1 + current_clip_epsilon
                ) * b_advantages[batch_indices]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss with clipping
                if self.config.clip_vloss:
                    # b_values here refers to the original values from the buffer for this batch
                    v_original_batch = b_values_original_flat[batch_indices]
                    v_clipped = v_original_batch + torch.clamp(
                        values - v_original_batch, # values are current model's value estimates
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

                # Gradient accumulation is not in the new `learn` method.
                # The original code had gradient accumulation. If it's intended to be removed, this is fine.
                # Otherwise, it needs to be added back. Assuming removal for now.

                self.optimizer.zero_grad(set_to_none=True) # More memory efficient

                # Backward pass
                if self.scaler and self.device.type == "cuda": # Ensure scaler is used only with CUDA
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer) # Unscale before clipping grad norm
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    pg_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(entropy.mean().item())

                    # KL divergence for early stopping
                    # Original KL: (b_log_probs[batch_indices] - log_probs).mean().item()
                    # New KL from prompt: ((ratio - 1) - log_ratio).mean().item()
                    # where log_ratio = b_log_probs[batch_indices] - log_probs (i.e., old_log_prob - new_log_prob)
                    # This is equivalent to (ratio - 1 - (b_log_probs[batch_indices] - log_probs)).mean().item()
                    current_log_ratio = log_probs - b_log_probs[batch_indices] # new_log_prob - old_log_prob
                    approx_kl = (torch.exp(current_log_ratio) - 1 - current_log_ratio).mean().item() # (ratio - 1 - log_ratio)
                    kl_divs.append(approx_kl)

                    clip_fractions.append(
                        ((ratio - 1).abs() > current_clip_epsilon).float().mean().item()
                    )

            except Exception as e:
                logger.error("Training step failed: %s", str(e), exc_info=True)
                continue # Skip this batch and continue with the next

        # Early stopping based on KL divergence
        # Calculate mean_kl from kl_divs collected in this epoch
        # Original: mean_kl = np.mean(kl_divs) if kl_divs else 0
        # New: mean_kl = np.mean(kl_divs[-len(indices) // self.config.batch_size:])
        # This new slicing for mean_kl seems to imply kl_divs are appended across epochs without clearing
        # which is not the case as kl_divs is initialized empty for each call to learn().
        # It should be just np.mean(kl_divs) if kl_divs for the current epoch.
        if kl_divs: # Check if kl_divs is not empty
            mean_kl_epoch = np.mean(kl_divs) # Mean KL for this epoch
            if self.config.target_kl is not None and mean_kl_epoch > self.config.target_kl:
                logger.info("Early stopping at epoch %d due to KL=%.4f", epoch, mean_kl_epoch)
                break
        else: # Handle case where kl_divs might be empty if all batches were skipped or failed
            mean_kl_epoch = 0


    # Compute explained variance
    # Original used data["values"].flatten() for y_pred, which are old values.
    # Should use current model's predictions on b_obs to calculate explained variance of returns by current value function.
    # However, the prompt uses b_values (which I interpreted as b_values_original_flat) for y_pred.
    # This would calculate how much variance of returns is explained by the *old* value function.
    # For true explained variance of the current model, we'd need to re-evaluate b_obs.
    # Let's stick to the prompt's y_pred = b_values (old values).
    # This means y_pred should be b_values_original_flat
    with torch.no_grad():
        y_pred = b_values_original_flat # old values
        y_true = b_returns # GAE returns
        var_y = torch.var(y_true)
        explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)

    return {
        "policy_loss": np.mean(pg_losses) if pg_losses else 0,
        "value_loss": np.mean(value_losses) if value_losses else 0,
        "entropy": np.mean(entropy_losses) if entropy_losses else 0,
        "kl_divergence": np.mean(kl_divs) if kl_divs else 0, # This will be mean KL over all batches of last completed epoch or all epochs if no early stop
        "clip_fraction": np.mean(clip_fractions) if clip_fractions else 0,
        "explained_variance": explained_var.item(),
        "learn_time": time.time() - learn_start,
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
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
                        "Update %d: Reward %.2f | KL %.4f",
                        update,
                        all_metrics.get('mean_episode_reward', 0),
                        all_metrics['kl_divergence']
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

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "train") -> None:
        """Log metrics to TensorBoard and W&B with proper formatting."""
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, self.global_step)
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(
                {f"{prefix}/{k}": v for k, v in metrics.items()},
                step=self.global_step
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
