# janus/training/ppo/trainer.py
"""
PPO Trainer for Proximal Policy Optimization (PPO) in JanusAI V2.

This module implements a production-ready PPO trainer that manages the training loop,
handles rollouts, normalizes rewards, and integrates with logging systems like 
TensorBoard and Weights & Biases. It is designed to be robust, configurable, 
and suitable for production-scale research.

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

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union, Any
from concurrent.futures import ThreadPoolExecutor
from torch.cuda.amp import GradScaler, autocast
from dataclasses import asdict

from .config import PPOConfig
from .buffer import RolloutBuffer
from .normalization import RunningMeanStd
from .collector import AsyncRolloutCollector
from .logging_utils import setup_logging, WANDB_AVAILABLE

if WANDB_AVAILABLE:
    import wandb

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    Production-ready PPO trainer with advanced features.
    
    This trainer orchestrates the entire PPO training process including:
    - Asynchronous environment stepping
    - Advantage estimation with GAE
    - Policy and value network updates
    - Hyperparameter scheduling
    - Comprehensive logging and checkpointing
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
        eval_env: Optional[Any] = None
    ):
        """
        Initialize the PPO trainer.
        
        Args:
            agent: PPO agent with actor and critic networks
            envs: List of training environments
            config: PPO configuration object
            experiment_name: Name for the experiment
            checkpoint_dir: Directory for saving checkpoints
            use_tensorboard: Whether to use TensorBoard logging
            use_wandb: Whether to use Weights & Biases logging
            eval_env: Optional environment for evaluation
        """
        self.agent = agent
        self.config = config
        self.eval_env = eval_env
        self.experiment_name = experiment_name
        
        # Device setup
        self.device = torch.device(config.device)
        self.agent.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Environment setup
        self.envs = envs
        self.n_envs = len(envs)
        
        # Get environment specifications
        dummy_obs, _ = envs[0].reset()
        self.obs_shape = dummy_obs.shape
        
        # Detect action space type
        if hasattr(envs[0].action_space, 'n'):
            # Discrete action space
            self.action_shape = ()
            self.action_dim = envs[0].action_space.n
            self.continuous_actions = False
        else:
            # Continuous action space
            self.action_shape = envs[0].action_space.shape
            self.action_dim = envs[0].action_space.shape[0]
            self.continuous_actions = True
            
        logger.info(f"Environment: obs_shape={self.obs_shape}, action_dim={self.action_dim}, "
                   f"continuous={self.continuous_actions}")
        
        # Async rollout collector
        self.collector = AsyncRolloutCollector(
            envs, self.device, config.num_workers, config.use_subprocess_envs
        )
        
        # Thread pool for async operations
        if config.num_workers > 1:
            self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        else:
            self.executor = None
            
        # Buffer will be initialized in train()
        self.buffer = None
        
        # Reward normalization
        self.reward_normalizer = RunningMeanStd() if config.normalize_rewards else None
        
        # Optimizer setup
        self.optimizer = torch.optim.Adam(
            agent.parameters(), 
            lr=config.learning_rate, 
            eps=1e-5
        )
        
        # Mixed precision setup
        self.scaler = GradScaler() if config.use_mixed_precision and self.device.type == 'cuda' else None
        
        # Training state
        self.global_step = 0
        self.num_updates = 0
        self.best_mean_reward = -np.inf
        self.best_eval_reward = -np.inf
        
        # Setup logging and checkpointing
        self.checkpoint_dir = checkpoint_dir or Path(f"checkpoints/{experiment_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save initial config
        config.to_yaml(self.checkpoint_dir / "config.yaml")
        
        # Setup loggers
        self.loggers = setup_logging(
            experiment_name, 
            self.checkpoint_dir, 
            use_tensorboard, 
            use_wandb, 
            asdict(config)
        )
        self.writer = self.loggers["writer"]
        self.csv_logger = self.loggers["csv"]
        
        logger.info(f"PPOTrainer initialized with {self.n_envs} environments, "
                   f"{config.num_workers} workers")
        
    def _schedule_hyperparam(
        self, 
        initial_value: float, 
        end_value: float,
        schedule: str, 
        progress: float
    ) -> float:
        """
        Calculate scheduled hyperparameter value.
        
        Args:
            initial_value: Starting value
            end_value: Final value
            schedule: Schedule type ('constant', 'linear', 'cosine')
            progress: Training progress [0, 1]
            
        Returns:
            Scheduled value
        """
        if schedule == 'constant':
            return initial_value
        elif schedule == 'linear':
            return initial_value + (end_value - initial_value) * progress
        elif schedule == 'cosine':
            return end_value + (initial_value - end_value) * 0.5 * (1 + np.cos(np.pi * progress))
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
            
    def collect_rollouts(self, num_steps: int) -> Dict[str, float]:
        """
        Collect rollouts from environments.
        
        Args:
            num_steps: Number of steps to collect per environment
            
        Returns:
            Dictionary of collection metrics
        """
        self.buffer.clear()
        collection_start = time.time()
        
        for step in range(num_steps):
            # Get current observations
            obs_tensor = torch.FloatTensor(self.collector.current_obs).to(self.device)
            
            with torch.no_grad():
                # Get actions and values from agent
                if self.scaler:
                    with autocast():
                        actions, log_probs = self.agent.act(obs_tensor)
                        values = self.agent.critic(obs_tensor).squeeze(-1)
                else:
                    actions, log_probs = self.agent.act(obs_tensor)
                    values = self.agent.critic(obs_tensor).squeeze(-1)
                    
            # Convert to numpy
            actions_np = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()
            values_np = values.cpu().numpy()
            
            # Store current observations
            current_obs = self.collector.current_obs.copy()
            
            # Step environments (async if executor available)
            next_obs, rewards, dones, truncateds = self.collector.collect_steps(
                actions_np, self.executor
            )
            
            # Apply reward normalization if configured
            if self.reward_normalizer is not None:
                self.reward_normalizer.update(rewards.reshape(-1, 1))
                normalized_rewards = rewards / np.sqrt(self.reward_normalizer.var + 1e-8)
            else:
                normalized_rewards = rewards
                
            # Store in buffer
            self.buffer.add(
                obs=current_obs,
                action=actions_np,
                reward=normalized_rewards,
                done=dones,
                value=values_np,
                log_prob=log_probs_np
            )
            
            self.global_step += self.n_envs
            
        collection_time = time.time() - collection_start
        
        # Get statistics
        stats = self.collector.get_statistics()
        stats['collection_time'] = collection_time
        stats['steps_per_second'] = (num_steps * self.n_envs) / collection_time
        
        return stats
        
    def compute_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor,
        dones: torch.Tensor, 
        last_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Rewards tensor [n_steps, n_envs]
            values: Value estimates [n_steps, n_envs]
            dones: Done flags [n_steps, n_envs]
            last_values: Bootstrap values [n_envs]
            
        Returns:
            advantages: GAE advantages
            returns: Target values for value function
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
                
            delta = rewards[t] + self.config.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = (
                delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae_lam
            )
            
        returns = advantages + values
        return advantages, returns
        
    def learn(
        self, 
        current_clip_epsilon: float, 
        current_entropy_coef: float
    ) -> Dict[str, float]:
        """
        Update policy and value networks using PPO.
        
        Args:
            current_clip_epsilon: Current clipping parameter
            current_entropy_coef: Current entropy coefficient
            
        Returns:
            Dictionary of training metrics
        """
        learn_start = time.time()
        data = self.buffer.get()
        
        # Compute advantages
        with torch.no_grad():
            # Get values for last observation (for bootstrapping)
            last_obs = torch.FloatTensor(self.collector.current_obs).to(self.device)
            if self.scaler:
                with autocast():
                    last_values = self.agent.critic(last_obs).squeeze(-1)
            else:
                last_values = self.agent.critic(last_obs).squeeze(-1)
                
            advantages, returns = self.compute_gae(
                data['rewards'], data['values'], data['dones'], last_values
            )
            
        # Flatten batch dimensions for training
        def flatten(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.swapaxes(0, 1).reshape(-1, *tensor.shape[2:])
            
        b_obs = flatten(data['observations'])
        b_actions = flatten(data['actions'])
        b_log_probs = flatten(data['log_probs']).squeeze()
        b_advantages = flatten(advantages).squeeze()
        b_returns = flatten(returns).squeeze()
        
        # Normalize advantages
        if self.config.normalize_advantages:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
        # Training metrics
        pg_losses, value_losses, entropy_losses, kl_divs, clip_fractions = [], [], [], [], []
        
        # Calculate total batch size considering gradient accumulation
        total_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps
        
        # Training epochs
        for epoch in range(self.config.n_epochs):
            # Shuffle indices for each epoch
            indices = np.random.permutation(b_obs.shape[0])
            
            # Gradient accumulation counter
            accumulation_counter = 0
            
            for start_idx in range(0, b_obs.shape[0], self.config.batch_size):
                batch_indices = indices[start_idx : start_idx + self.config.batch_size]
                
                # Skip incomplete batches
                if len(batch_indices) < self.config.batch_size // 2:
                    continue
                    
                # Forward pass
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
                    
                    # PPO losses
                    ratio = torch.exp(log_probs - b_log_probs[batch_indices])
                    surr1 = ratio * b_advantages[batch_indices]
                    surr2 = torch.clamp(
                        ratio, 
                        1 - current_clip_epsilon, 
                        1 + current_clip_epsilon
                    ) * b_advantages[batch_indices]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss (consider clipped value loss)
                    value_pred_clipped = data['values'].flatten()[batch_indices] + torch.clamp(
                        values - data['values'].flatten()[batch_indices],
                        -current_clip_epsilon,
                        current_clip_epsilon
                    )
                    value_losses_clipped = (value_pred_clipped - b_returns[batch_indices]) ** 2
                    value_losses_unclipped = (values - b_returns[batch_indices]) ** 2
                    value_loss = 0.5 * torch.max(value_losses_clipped, value_losses_unclipped).mean()
                    
                    # Total loss
                    loss = (
                        policy_loss + 
                        self.config.value_coef * value_loss - 
                        current_entropy_coef * entropy.mean()
                    )
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                    
                    # Backward pass
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                        
                    accumulation_counter += 1
                    
                    # Optimizer step after accumulation
                    if accumulation_counter % self.config.gradient_accumulation_steps == 0:
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(
                                self.agent.parameters(), 
                                self.config.max_grad_norm
                            )
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            nn.utils.clip_grad_norm_(
                                self.agent.parameters(), 
                                self.config.max_grad_norm
                            )
                            self.optimizer.step()
                            
                        self.optimizer.zero_grad()
                        
                    # Track metrics
                    with torch.no_grad():
                        pg_losses.append(policy_loss.item() * self.config.gradient_accumulation_steps)
                        value_losses.append(value_loss.item())
                        entropy_losses.append(entropy.mean().item())
                        kl_div = (b_log_probs[batch_indices] - log_probs).mean().item()
                        kl_divs.append(kl_div)
                        clip_fraction = (torch.abs(ratio - 1) > current_clip_epsilon).float().mean().item()
                        clip_fractions.append(clip_fraction)
                        
                except Exception as e:
                    logger.error(f"Error in training step: {e}", exc_info=True)
                    continue
                    
            # Early stopping based on KL divergence
            mean_kl = np.mean(kl_divs) if kl_divs else 0
            if self.config.target_kl is not None and mean_kl > self.config.target_kl:
                logger.info(f"Early stopping at epoch {epoch+1} due to high KL divergence: {mean_kl:.4f}")
                break
                
        # Compute explained variance
        with torch.no_grad():
            y_pred = data['values'].flatten()
            y_true = returns.flatten()
            var_y = torch.var(y_true)
            explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
            
        learn_time = time.time() - learn_start
        
        metrics = {
            'policy_loss': np.mean(pg_losses) if pg_losses else 0,
            'value_loss': np.mean(value_losses) if value_losses else 0,
            'entropy': np.mean(entropy_losses) if entropy_losses else 0,
            'kl_divergence': np.mean(kl_divs) if kl_divs else 0,
            'clip_fraction': np.mean(clip_fractions) if clip_fractions else 0,
            'explained_variance': explained_var.item(),
            'learn_time': learn_time,
            'epochs_trained': epoch + 1,
        }
        
        return metrics
        
    def evaluate(
        self, 
        eval_env: Optional[Any] = None, 
        num_episodes: int = 10,
        render: bool = False
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
            return {'eval_reward': 0.0}
            
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                if render:
                    env.render()
                    
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, _ = self.agent.act(obs_tensor, deterministic=True)
                    
                if self.continuous_actions:
                    action_np = action.cpu().numpy()[0]
                else:
                    action_np = action.cpu().numpy()[0].item()
                    
                obs, reward, done, truncated, _ = env.step(action_np)
                episode_reward += reward
                episode_length += 1
                
                if truncated:
                    done = True
                    
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        return {
            'eval_reward': np.mean(episode_rewards),
            'eval_reward_std': np.std(episode_rewards),
            'eval_length': np.mean(episode_lengths),
        }
        
    def save_checkpoint(self, path: Optional[Path] = None, is_best: bool = False) -> Path:
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint (auto-generated if None)
            is_best: Whether this is the best model so far
            
        Returns:
            Path where checkpoint was saved
        """
        if path is None:
            if is_best:
                path = self.checkpoint_dir / "best_model.pt"
            else:
                path = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
                
        checkpoint = {
            'global_step': self.global_step,
            'num_updates': self.num_updates,
            'best_mean_reward': self.best_mean_reward,
            'best_eval_reward': self.best_eval_reward,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
        }
        
        # Add optional components
        if self.reward_normalizer is not None:
            checkpoint['reward_normalizer'] = self.reward_normalizer
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        torch.save(checkpoint, path)
        logger.info(f"{'Best ' if is_best else ''}Checkpoint saved to {path}")
        
        return path
        
    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.num_updates = checkpoint.get('num_updates', 0)
        self.best_mean_reward = checkpoint.get('best_mean_reward', -np.inf)
        self.best_eval_reward = checkpoint.get('best_eval_reward', -np.inf)
        
        if 'reward_normalizer' in checkpoint and self.config.normalize_rewards:
            self.reward_normalizer = checkpoint['reward_normalizer']
            
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        logger.info(f"Checkpoint loaded from {path} (step {self.global_step})")
        
        return checkpoint
        
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "train"):
        """
        Log metrics to all configured backends.
        
        Args:
            metrics: Dictionary of metrics to log
            prefix: Prefix for metric names
        """
        # TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{prefix}/{key}", value, self.global_step)
                
        # Weights & Biases
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            wandb_metrics['global_step'] = self.global_step
            wandb.log(wandb_metrics)
            
        # CSV logging for training metrics
        if prefix == "train" and self.csv_logger is not None:
            self.csv_logger.info(
                f"{self.num_updates},{self.global_step},"
                f"{metrics.get('mean_episode_reward', 0):.2f},"
                f"{metrics.get('policy_loss', 0):.4f},"
                f"{metrics.get('value_loss', 0):.4f},"
                f"{metrics.get('entropy', 0):.4f},"
                f"{metrics.get('kl_divergence', 0):.4f},"
                f"{metrics.get('learning_rate', 0):.6f}"
            )
            
    def train(self, total_timesteps: int, rollout_length: int = 2048):
        """
        Main training loop.
        
        Args:
            total_timesteps: Total environment steps to train for
            rollout_length: Steps to collect per rollout
        """
        # Initialize buffer
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            device=self.device,
            n_envs=self.n_envs
        )
        
        num_updates = total_timesteps // (rollout_length * self.n_envs)
        
        logger.info(f"Starting training for {total_timesteps} timesteps ({num_updates} updates)")
        logger.info(f"Rollout length: {rollout_length}, Environments: {self.n_envs}")
        
        start_time = time.time()
        
        try:
            for update in range(1, num_updates + 1):
                self.num_updates = update
                update_start_time = time.time()
                progress = update / num_updates
                
                # Schedule hyperparameters
                lr = self._schedule_hyperparam(
                    self.config.learning_rate, 
                    self.config.lr_end,
                    self.config.lr_schedule, 
                    progress
                )
                clip_epsilon = self._schedule_hyperparam(
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
                
                # Update learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                    
                # Collect rollouts
                collection_metrics = self.collect_rollouts(rollout_length)
                
                # Learn from collected data
                learning_metrics = self.learn(clip_epsilon, entropy_coef)
                
                # Combine metrics
                update_time = time.time() - update_start_time
                all_metrics = {
                    **collection_metrics,
                    **learning_metrics,
                    'learning_rate': lr,
                    'clip_epsilon': clip_epsilon,
                    'entropy_coef': entropy_coef,
                    'update_time': update_time,
                    'total_time': time.time() - start_time,
                }
                
                # Logging
                if update % self.config.log_interval == 0:
                    logger.info(
                        f"Update {update}/{num_updates} | "
                        f"Steps: {self.global_step} | "
                        f"Reward: {all_metrics['mean_episode_reward']:.2f} | "
                        f"Policy Loss: {all_metrics['policy_loss']:.4f} | "
                        f"Value Loss: {all_metrics['value_loss']:.4f} | "
                        f"Entropy: {all_metrics['entropy']:.4f} | "
                        f"KL: {all_metrics['kl_divergence']:.4f} | "
                        f"Clip: {all_metrics['clip_fraction']:.3f} | "
                        f"SPS: {all_metrics['steps_per_second']:.0f}"
                    )
                    self._log_metrics(all_metrics)
                    
                # Evaluation
                if self.eval_env is not None and update % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate(num_episodes=10)
                    logger.info(
                        f"Evaluation: {eval_metrics['eval_reward']:.2f} ± "
                        f"{eval_metrics['eval_reward_std']:.2f}"
                    )
                    self._log_metrics(eval_metrics, prefix="eval")
                    
                    # Track best eval model
                    if eval_metrics['eval_reward'] > self.best_eval_reward:
                        self.best_eval_reward = eval_metrics['eval_reward']
                        self.save_checkpoint(is_best=True)
                        
                # Checkpointing
                if all_metrics['mean_episode_reward'] > self.best_mean_reward:
                    self.best_mean_reward = all_metrics['mean_episode_reward']
                    if self.eval_env is None:  # Save best if no eval env
                        self.save_checkpoint(is_best=True)
                        
                if update % self.config.save_interval == 0:
                    self.save_checkpoint()
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with exception: {e}", exc_info=True)
            raise
        finally:
            # Cleanup
            if self.executor is not None:
                self.executor.shutdown(wait=True)
                
            # Save final model
            self.save_checkpoint(self.checkpoint_dir / "final_model.pt")
            
            # Close logging
            if self.writer is not None:
                self.writer.close()
                
            total_time = time.time() - start_time
            logger.info(f"Training completed! Total time: {total_time/3600:.2f} hours")
            logger.info(f"Best mean reward: {self.best_mean_reward:.2f}")
            if self.eval_env is not None:
                logger.info(f"Best eval reward: {self.best_eval_reward:.2f}")