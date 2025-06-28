# File: janus/training/ppo/main.py
"""
Proximal Policy Optimization (PPO) Training Script for JanusAI V2

This script orchestrates the PPO training process by breaking it down into
modular components for configuration, environment setup, agent creation,
and training execution.

Example usage:
    python main.py --config demo_config.yaml --num-envs 8 --seed 42
"""

import logging
import argparse
from pathlib import Path
import sys
import torch
import numpy as np
import random
from typing import Tuple, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from janus.training.ppo.config import PPOConfig
from janus.training.ppo.trainer import PPOTrainer
from janus.agents.ppo_agent import PPOAgent, NetworkConfig
from janus.envs.symbolic_regression import SymbolicRegressionEnv
from janus.core.base_env import BaseEnv

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Sets the random seed for reproducibility across all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


def setup_configuration(args: argparse.Namespace) -> PPOConfig:
    """Loads PPO configuration and overrides with command-line arguments."""
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = PPOConfig.from_yaml(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    # Override with command-line arguments
    if args.num_envs is not None:
        config.num_workers = args.num_envs
    if args.device is not None:
        config.device = args.device
    return config


def setup_environments(num_envs: int, enable_eval: bool) -> Tuple[List[BaseEnv], Optional[BaseEnv]]:
    """Creates training and evaluation environments."""
    logger.info(f"Creating {num_envs} parallel environments...")
    envs = [SymbolicRegressionEnv() for _ in range(num_envs)]
    eval_env = SymbolicRegressionEnv() if enable_eval else None
    return envs, eval_env


def setup_agent(config: PPOConfig, args: argparse.Namespace, envs: List[BaseEnv]) -> PPOAgent:
    """Prepares network configs and creates the PPO agent."""
    # Get environment specifications
    obs_dim = envs[0].observation_space.shape[0]
    if hasattr(envs[0].action_space, 'n'):
        action_dim = envs[0].action_space.n
        continuous_actions = False
    else:
        action_dim = envs[0].action_space.shape[0]
        continuous_actions = True
    logger.info(f"Env specs: obs_dim={obs_dim}, action_dim={action_dim}, continuous={continuous_actions}")

    # Load network configurations
    config_dir = Path(args.config).parent
    actor_config = NetworkConfig.from_yaml(config_dir / "actor_network_config.yaml")
    critic_config = NetworkConfig.from_yaml(config_dir / "critic_network_config.yaml")

    logger.info("Creating PPO agent...")
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
    
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    return agent


def run_training(trainer: PPOTrainer, args: argparse.Namespace):
    """Handles checkpoint loading and the main training loop."""
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            trainer.load_checkpoint(checkpoint_path)
        else:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return 1  # Return error code

    try:
        logger.info("Starting training...")
        trainer.train(
            total_timesteps=args.total_timesteps,
            rollout_length=args.rollout_length
        )
        logger.info("Training completed successfully!")
        return 0  # Return success code
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user. Shutting down...")
        return 0
    except Exception as e:
        logger.error(f"Training failed with an unexpected error: {e}", exc_info=True)
        return 1


def main(cli_args: argparse.Namespace) -> int:
    """Main training orchestration function."""
    try:
        # Step 1: Set seed for reproducibility
        if cli_args.seed is not None:
            set_seed(cli_args.seed)

        # Step 2: Load configurations
        config = setup_configuration(cli_args)

        # Step 3: Create environments
        envs, eval_env = setup_environments(config.num_workers, cli_args.eval)

        # Step 4: Create agent
        agent = setup_agent(config, cli_args, envs)

        # Step 5: Create trainer
        logger.info("Creating PPO trainer...")
        trainer = PPOTrainer(
            agent=agent,
            envs=envs,
            config=config,
            experiment_name=cli_args.experiment_name,
            checkpoint_dir=Path(cli_args.checkpoint_dir) if cli_args.checkpoint_dir else None,
            use_tensorboard=cli_args.tensorboard,
            use_wandb=cli_args.wandb,
            eval_env=eval_env,
        )

        # Step 6: Run the training loop
        return run_training(trainer, cli_args)

    except Exception as e:
        logger.error(f"A critical error occurred in the setup phase: {e}", exc_info=True)
        return 1


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PPO Training for JanusAI V2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # ... (Argument definitions remain the same)
    parser.add_argument("--config", type=str, default="configs/demo_config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total environment steps to train for")
    parser.add_argument("--rollout-length", type=int, default=2048, help="Number of steps per rollout")
    parser.add_argument("--num-envs", type=int, default=None, help="Number of parallel environments (overrides config)")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default=None, help="Device for training (overrides config)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-name", type=str, default="ppo_experiment", help="Name for this experiment")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--eval", action="store_true", help="Enable evaluation during training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    exit_code = main(args)
    sys.exit(exit_code)