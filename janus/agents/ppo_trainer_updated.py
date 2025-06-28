# janus/agents/ppo_trainer_updated.py
"""Updated PPOTrainer with modular reward system integration."""

from typing import Any, Dict, Optional, Tuple
import torch
import numpy as np
from janus.rewards import RewardHandler


class PPOTrainerUpdated:
    """PPO Trainer with integrated RewardHandler.
    
    This shows the key modifications needed to integrate
    the modular reward system into your existing trainer.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize PPO trainer with reward handler.
        
        Args:
            config: Full configuration dictionary.
        """
        # ... existing initialization code ...
        
        # Initialize reward handler from config
        reward_config = config.get('rewards', {})
        self.reward_handler = RewardHandler(reward_config)
        
        # Optional: enable reward debugging
        self.debug_rewards = config.get('debug_rewards', False)
        
    def collect_rollout_step(
        self,
        obs: np.ndarray,
        env_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Collect a single rollout step with modular rewards.
        
        Args:
            obs: Current observation.
            env_info: Optional environment information.
            
        Returns:
            Tuple of (next_obs, reward, done, info).
        """
        # Get action from policy
        action = self.policy.get_action(obs)
        
        # Step environment
        next_obs, env_reward, done, info = self.env.step(action)
        
        # Compute modular reward
        modular_reward = self.reward_handler.compute_reward(
            observation=obs,
            action=action,
            next_observation=next_obs,
            info=info
        )
        
        # Log reward breakdown if debugging
        if self.debug_rewards:
            breakdown = self.reward_handler.get_component_breakdown(
                obs, action, next_obs, info
            )
            self._log_reward_breakdown(breakdown)
            
        # Store transition with modular reward
        self.buffer.add(
            obs=obs,
            action=action,
            reward=modular_reward,  # Use modular reward
            next_obs=next_obs,
            done=done,
            info=info
        )
        
        return next_obs, modular_reward, done, info
    
    def _log_reward_breakdown(
        self,
        breakdown: Dict[str, float]
    ) -> None:
        """Log individual reward components for debugging.
        
        Args:
            breakdown: Dictionary of component rewards.
        """
        log_str = "Reward breakdown: "
        for component, value in breakdown.items():
            if not component.endswith('_weighted'):
                log_str += f"{component}={value:.3f}, "
        log_str += f"total={breakdown['total']:.3f}"
        
        if hasattr(self, 'logger'):
            self.logger.debug(log_str)
        else:
            print(log_str)
            
    def collect_rollouts(self, n_steps: int) -> None:
        """Collect multiple rollout steps.
        
        Args:
            n_steps: Number of steps to collect.
        """
        obs = self.env.reset()
        
        for _ in range(n_steps):
            obs, reward, done, info = self.collect_rollout_step(obs)
            
            if done:
                obs = self.env.reset()
                # Reset reward components at episode boundary
                self.reward_handler.reset()
                
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with reward tracking.
        
        Returns:
            Dictionary of training metrics.
        """
        # Collect rollouts with modular rewards
        self.collect_rollouts(self.rollout_length)
        
        # ... existing training code ...
        
        # Add reward statistics to metrics
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'mean_reward': self.buffer.rewards.mean(),
            # ... other metrics ...
        }
        
        return metrics


# Example configuration file update
"""
# config.yaml - Add this rewards section

training:
  n_epochs: 1000
  batch_size: 64
  learning_rate: 3e-4
  rollout_length: 2048
  debug_rewards: true  # Enable reward debugging

rewards:
  components:
    # Primary task completion reward
    - type: task_success
      enabled: true
      weight: 10.0
      success_reward: 100.0
      
    # Penalize overly complex solutions
    - type: complexity_penalty
      enabled: true
      weight: 0.1
      penalty_weight: 1.0
      
    # Encourage exploration
    - type: novelty_bonus
      enabled: true
      weight: 0.5
      bonus_scale: 2.0
      memory_size: 1000
      
    # Reward incremental progress
    - type: progress
      enabled: true
      weight: 1.0
      scale_factor: 5.0
"""


# scripts/train_with_modular_rewards.py
"""Training script with modular reward system."""

import yaml
from pathlib import Path
from janus.agents import PPOTrainerUpdated
from janus.utils.logging import setup_logger


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main() -> None:
    """Main training loop with modular rewards."""
    # Load configuration
    config = load_config(Path('configs/config.yaml'))
    
    # Setup logging
    logger = setup_logger('training', config['logging'])
    
    # Initialize trainer with reward handler
    trainer = PPOTrainerUpdated(config)
    
    # Training loop
    for epoch in range(config['training']['n_epochs']):
        metrics = trainer.train_epoch()
        
        # Log metrics
        logger.info(
            f"Epoch {epoch}: "
            f"reward={metrics['mean_reward']:.2f}, "
            f"policy_loss={metrics['policy_loss']:.4f}"
        )
        
        # Save checkpoint periodically
        if epoch % config['training']['save_interval'] == 0:
            trainer.save_checkpoint(
                Path(f"results/checkpoints/epoch_{epoch}.pt")
            )
            
    logger.info("Training completed!")


if __name__ == "__main__":
    main()