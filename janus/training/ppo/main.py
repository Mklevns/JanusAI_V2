# janus/training/ppo/main.py
"""
Proximal Policy Optimization (PPO) Training Script for JanusAI V2

This script demonstrates how to set up and run a PPO training session using 
the JanusAI V2 framework with proper network configurations and error handling.

Example usage:
    python main.py --config demo_config.yaml --num-envs 8 --total-timesteps 1000000
"""

import logging
import argparse
from pathlib import Path
from typing import List
import sys
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from janus.training.ppo.config import PPOConfig
from janus.training.ppo.trainer import PPOTrainer
from janus.agents.ppo_agent import PPOAgent, NetworkConfig
from janus.envs.symbolic_regression import SymbolicRegressionEnv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)


def make_env():
    """Factory function to create environment instances."""
    return SymbolicRegressionEnv()


def load_network_config(config_path: Path) -> NetworkConfig:
    """Load network configuration from YAML file."""
    try:
        return NetworkConfig.from_yaml(config_path)
    except FileNotFoundError:
        logger.warning(f"Network config not found at {config_path}, using defaults")
        return NetworkConfig(
            layer_sizes=[256, 256],
            activation="tanh",
            initialization="orthogonal"
        )


def main(args):
    """Main training function."""
    logger.info("=== Starting PPO Training ===")
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1
        
    try:
        config = PPOConfig.from_yaml(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1
    
    # Override config with command line arguments
    if args.num_envs:
        config.num_workers = args.num_envs
    if args.device:
        config.device = args.device
        
    # Create environments
    logger.info(f"Creating {config.num_workers} environments...")
    try:
        envs = [make_env() for _ in range(config.num_workers)]
        eval_env = make_env() if args.eval else None
    except Exception as e:
        logger.error(f"Failed to create environments: {e}")
        return 1
    
    # Get environment specifications
    obs_dim = envs[0].observation_space.shape[0]
    if hasattr(envs[0].action_space, 'n'):
        action_dim = envs[0].action_space.n
        continuous_actions = False
    else:
        action_dim = envs[0].action_space.shape[0]
        continuous_actions = True
        
    logger.info(f"Environment: obs_dim={obs_dim}, action_dim={action_dim}, "
               f"continuous={continuous_actions}")
    
    # Load network configurations
    config_dir = config_path.parent
    actor_config = load_network_config(config_dir / "actor_network_config.yaml")
    critic_config = load_network_config(config_dir / "critic_network_config.yaml")
    
    # Create agent
    logger.info("Creating PPO agent...")
    try:
        agent = PPOAgent(
            observation_dim=obs_dim,
            action_dim=action_dim,
            actor_config=actor_config,
            critic_config=critic_config,
            continuous_actions=continuous_actions,
            device=config.device,
            seed=args.seed,
            enable_amp=config.use_mixed_precision
        )
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        return 1
    
    # Log model architecture
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create trainer
    logger.info("Creating PPO trainer...")
    trainer = PPOTrainer(
        agent=agent,
        envs=envs,
        config=config,
        experiment_name=args.experiment_name,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        use_tensorboard=args.tensorboard,
        use_wandb=args.wandb,
        eval_env=eval_env,
    )
    
    # Load checkpoint if specified
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            trainer.load_checkpoint(checkpoint_path)
        else:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return 1
    
    # Start training
    try:
        logger.info("Starting training...")
        trainer.train(
            total_timesteps=args.total_timesteps,
            rollout_length=args.rollout_length
        )
        logger.info("Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PPO Training for JanusAI V2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        default="demo_config.yaml",
        help="Path to configuration YAML file"
    )
    
    # Training parameters
    parser.add_argument(
        "--total-timesteps", 
        type=int, 
        default=1_000_000,
        help="Total environment steps to train for"
    )
    parser.add_argument(
        "--rollout-length", 
        type=int, 
        default=2048,
        help="Number of steps per rollout"
    )
    parser.add_argument(
        "--num-envs", 
        type=int, 
        default=None,
        help="Number of parallel environments (overrides config)"
    )
    
    # Hardware
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Device to use for training (overrides config)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducibility"
    )
    
    # Experiment settings
    parser.add_argument(
        "--experiment-name", 
        type=str, 
        default="ppo_experiment",
        help="Name for this experiment"
    )
    parser.add_argument(
        "--checkpoint-dir", 
        type=str, 
        default=None,
        help="Directory for saving checkpoints"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    # Logging
    parser.add_argument(
        "--tensorboard", 
        action="store_true",
        help="Enable TensorBoard logging"
    )
    parser.add_argument(
        "--wandb", 
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--eval", 
        action="store_true",
        help="Enable evaluation during training"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Set random seed if specified
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Set random seed to {args.seed}")
    
    # Run training
    exit_code = main(args)
    sys.exit(exit_code)