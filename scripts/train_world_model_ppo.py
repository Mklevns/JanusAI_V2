# scripts/train_world_model_ppo.py
"""
Training script for World Model PPO.

This script demonstrates how to train a PPO agent using the World Model
approach, where the agent learns from both real and imagined experience.

Usage:
    python scripts/train_world_model_ppo.py --env CartPole-v1 --total-timesteps 1000000
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import gym
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from janus.agents.ppo_agent import PPOAgent, NetworkConfig
from janus.training.ppo.config import PPOConfig
from janus.training.ppo.world_model_trainer import WorldModelPPOTrainer, WorldModelConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_world_model_config(config_path: Path) -> WorldModelConfig:
    """Load world model configuration from YAML."""
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)

    wm_config_dict = full_config.get('world_model', {})

    # Parse nested configuration
    config = WorldModelConfig(
        vae_latent_dim=wm_config_dict['vae']['latent_dim'],
        vae_hidden_dims=wm_config_dict['vae']['hidden_dims'],
        vae_beta=wm_config_dict['vae']['beta'],
        vae_lr=wm_config_dict['vae']['learning_rate'],
        mdn_hidden_dim=wm_config_dict['mdn_rnn']['hidden_dim'],
        mdn_num_mixtures=wm_config_dict['mdn_rnn']['num_mixtures'],
        mdn_temperature=wm_config_dict['mdn_rnn']['temperature'],
        mdn_lr=wm_config_dict['mdn_rnn']['learning_rate'],
        pretrain_epochs=wm_config_dict['training']['pretrain_epochs'],
        pretrain_batch_size=wm_config_dict['training']['pretrain_batch_size'],
        imagination_ratio=wm_config_dict['training']['imagination_ratio'],
        imagination_horizon=wm_config_dict['training']['imagination_horizon'],
        random_collection_steps=wm_config_dict['data_collection']['random_collection_steps']
    )

    return config


def create_environments(env_name: str, num_envs: int, seed: int = 42) -> List[gym.Env]:
    """Create a list of environments."""
    envs = []
    for i in range(num_envs):
        env = gym.make(env_name)
        env.seed(seed + i)
        envs.append(env)
    return envs


def main(args):
    """Main training function."""
    # Load configurations
    config_path = Path(args.config)
    ppo_config = PPOConfig.from_yaml(config_path)
    wm_config = load_world_model_config(config_path)

    # Override with command line args
    if args.device:
        ppo_config.device = args.device

    logger.info(f"Loaded configuration from {config_path}")
    logger.info(f"Training on environment: {args.env}")
    logger.info(f"Total timesteps: {args.total_timesteps:,}")

    # Create environments
    envs = create_environments(args.env, args.num_envs, args.seed)
    eval_env = gym.make(args.env) if args.eval else None

    # Get environment info
    obs_dim = envs[0].observation_space.shape[0]
    if hasattr(envs[0].action_space, 'n'):
        action_dim = envs[0].action_space.n
        continuous = False
    else:
        action_dim = envs[0].action_space.shape[0]
        continuous = True

    logger.info(f"Environment: obs_dim={obs_dim}, action_dim={action_dim}, continuous={continuous}")

    # Create agent
    # Note: The actual input dim will be modified by the trainer
    agent = PPOAgent(
        observation_dim=obs_dim,
        action_dim=action_dim,
        actor_config=NetworkConfig(
            layer_sizes=[256, 128],
            activation="tanh",
            use_layer_norm=True
        ),
        critic_config=NetworkConfig(
            layer_sizes=[256, 128],
            activation="relu"
        ),
        continuous_actions=continuous,
        device=ppo_config.device,
        seed=args.seed
    )

    # Create trainer
    trainer = WorldModelPPOTrainer(
        agent=agent,
        envs=envs,
        config=ppo_config,
        world_model_config=wm_config,
        experiment_name=args.experiment_name,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        use_tensorboard=args.tensorboard,
        use_wandb=args.wandb,
        eval_env=eval_env
    )

    # Load checkpoint if resuming
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            trainer.load_checkpoint(checkpoint_path)
        else:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return 1

    # Print training summary
    logger.info("\n" + "="*50)
    logger.info("Training Summary:")
    logger.info(f"  Agent Architecture:")
    logger.info(f"    - Original obs dim: {obs_dim}")
    logger.info(f"    - Latent dim: {wm_config.vae_latent_dim}")
    logger.info(f"    - RNN hidden dim: {wm_config.mdn_hidden_dim}")
    logger.info(f"  World Model:")
    logger.info(f"    - VAE layers: {wm_config.vae_hidden_dims}")
    logger.info(f"    - MDN mixtures: {wm_config.mdn_num_mixtures}")
    logger.info(f"  Training:")
    logger.info(f"    - Imagination ratio: {wm_config.imagination_ratio*100:.0f}%")
    logger.info(f"    - Imagination horizon: {wm_config.imagination_horizon}")
    logger.info("="*50 + "\n")

    # Train
    try:
        trainer.train(
            total_timesteps=args.total_timesteps,
            rollout_length=args.rollout_length
        )
        logger.info("Training completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        # Save checkpoint before exiting
        trainer.save_checkpoint()
        return 0

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO with World Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Environment
    parser.add_argument(
        '--env',
        type=str,
        default='CartPole-v1',
        help='Gym environment ID'
    )
    parser.add_argument(
        '--num-envs',
        type=int,
        default=4,
        help='Number of parallel environments'
    )

    # Training
    parser.add_argument(
        '--config',
        type=str,
        default='configs/world_model_ppo.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--total-timesteps',
        type=int,
        default=1_000_000,
        help='Total environment steps to train'
    )
    parser.add_argument(
        '--rollout-length',
        type=int,
        default=2048,
        help='Number of steps per rollout'
    )

    # Hardware
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default=None,
        help='Device to use (overrides config)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    # Experiment
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='world_model_ppo',
        help='Name for this experiment'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory for saving checkpoints'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    # Logging
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='Enable TensorBoard logging'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Enable evaluation during training'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    exit_code = main(args)
    sys.exit(exit_code)
