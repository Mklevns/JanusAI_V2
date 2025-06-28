# File: janus/training/ppo/main.py
"""
Proximal Policy Optimization (PPO) Training Script for JanusAI V2

This script orchestrates the PPO training process by breaking it down into
modular components for configuration, environment setup, agent creation,
and training execution.

Example usage:
    python main.py --config demo_config.yaml --num-envs 8 --seed 42
"""

# Standard library imports
import argparse
import logging
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Third-party imports
import numpy as np
import torch

# Add parent directory to path for imports (before local imports)
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

# Local imports
from janus.agents.ppo_agent import NetworkConfig, PPOAgent
from janus.core.base_env import BaseEnv
from janus.envs.symbolic_regression import SymbolicRegressionEnv
from janus.training.ppo.config import PPOConfig
from janus.training.ppo.trainer import PPOTrainer

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Sets the random seed for reproducibility across all relevant libraries.

    Args:
        seed: Random seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Set random seed to %d", seed)


def setup_configuration(cmd_args: argparse.Namespace) -> PPOConfig:
    """
    Loads PPO configuration and overrides with command-line arguments.

    Args:
        cmd_args: Command line arguments

    Returns:
        PPOConfig instance with overrides applied
    """
    config_path = Path(cmd_args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = PPOConfig.from_yaml(config_path)
    logger.info("Loaded configuration from %s", config_path)

    # Override with command-line arguments
    if cmd_args.num_envs is not None:
        config.num_workers = cmd_args.num_envs
    if cmd_args.device is not None:
        config.device = cmd_args.device
    return config


def setup_environments(
    num_envs: int, enable_eval: bool, seed: int = 42
) -> Tuple[List[BaseEnv], Optional[BaseEnv]]:
    """
    Create environments with proper seeding.

    Args:
        num_envs: Number of parallel environments to create
        enable_eval: Whether to create an evaluation environment
        seed: Base seed for environment initialization

    Returns:
        Tuple of (training_envs, eval_env)
    """
    logger.info(
        "Creating %d parallel environments with base seed %d...", num_envs, seed
    )

    # Create environments with different seeds
    envs = []
    for i in range(num_envs):
        env = SymbolicRegressionEnv()
        env.seed(seed + i)
        envs.append(env)

    eval_env = None
    if enable_eval:
        eval_env = SymbolicRegressionEnv()
        eval_env.seed(seed + num_envs)  # Different seed for eval
        logger.info("Evaluation environment created with seed %d", seed + num_envs)

    return envs, eval_env


def setup_agent(
    config: PPOConfig, cmd_args: argparse.Namespace, envs: List[BaseEnv]
) -> PPOAgent:
    """
    Prepares network configs and creates the PPO agent.

    Args:
        config: PPO configuration
        cmd_args: Command line arguments
        envs: List of environments

    Returns:
        Initialized PPO agent
    """
    # Get environment specifications
    obs_dim = envs[0].observation_space.shape[0]
    if hasattr(envs[0].action_space, "n"):
        action_dim = envs[0].action_space.n
        continuous_actions = False
    else:
        action_dim = envs[0].action_space.shape[0]
        continuous_actions = True

    logger.info(
        "Env specs: obs_dim=%d, action_dim=%d, continuous=%s",
        obs_dim,
        action_dim,
        continuous_actions,
    )

    # Load network configurations
    config_dir = Path(cmd_args.config).parent
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
        seed=cmd_args.seed,
        enable_amp=config.use_mixed_precision,
    )

    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    logger.info(
        "Model parameters: %s total, %s trainable",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )
    return agent


def run_training(trainer: PPOTrainer, cmd_args: argparse.Namespace) -> int:
    """
    Handles checkpoint loading and the main training loop.

    Args:
        trainer: PPO trainer instance
        cmd_args: Command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    if cmd_args.resume:
        checkpoint_path = Path(cmd_args.resume)
        if checkpoint_path.exists():
            logger.info("Resuming from checkpoint: %s", checkpoint_path)
            trainer.load_checkpoint(checkpoint_path)
        else:
            logger.error("Checkpoint not found: %s", checkpoint_path)
            return 1  # Return error code

    try:
        logger.info("Starting training...")

        # Optional system resource check
        if hasattr(trainer, "check_system_resources"):
            trainer.check_system_resources()

        trainer.train(
            total_timesteps=cmd_args.total_timesteps,
            rollout_length=cmd_args.rollout_length,
        )
        logger.info("Training completed successfully!")
        return 0  # Return success code

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user. Shutting down...")
        return 0
    except (RuntimeError, ValueError, TypeError) as e:
        logger.error("Training failed with error: %s", str(e), exc_info=True)
        return 1


def main(cli_args: argparse.Namespace) -> int:
    """
    Main training orchestration function.

    Args:
        cli_args: Command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Step 1: Set seed for reproducibility
        if cli_args.seed is not None:
            set_seed(cli_args.seed)

        # Step 2: Load configurations
        config = setup_configuration(cli_args)

        # Step 3: Create environments
        envs, eval_env = setup_environments(
            config.num_workers, cli_args.eval, cli_args.seed
        )

        # Step 4: Create agent
        agent = setup_agent(config, cli_args, envs)

        # Step 5: Create trainer
        logger.info("Creating PPO trainer...")

        checkpoint_dir = None
        if cli_args.checkpoint_dir:
            checkpoint_dir = Path(cli_args.checkpoint_dir)

        trainer = PPOTrainer(
            agent=agent,
            envs=envs,
            config=config,
            experiment_name=cli_args.experiment_name,
            checkpoint_dir=checkpoint_dir,
            use_tensorboard=cli_args.tensorboard,
            use_wandb=cli_args.wandb,
            eval_env=eval_env,
        )

        # Step 6: Run the training loop
        return run_training(trainer, cli_args)

    except (FileNotFoundError, ValueError, TypeError) as e:
        logger.error(
            "A critical error occurred in the setup phase: %s", str(e), exc_info=True
        )
        return 1


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PPO Training for JanusAI V2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/demo_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total environment steps to train for",
    )
    parser.add_argument(
        "--rollout-length", type=int, default=2048, help="Number of steps per rollout"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of parallel environments (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Device for training (overrides config)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ppo_experiment",
        help="Name for this experiment",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging"
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--eval", action="store_true", help="Enable evaluation during training"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    EXIT_CODE = main(args)
    sys.exit(EXIT_CODE)
# End of janus/training/ppo/main.py
# This script is the entry point for training a PPO agent in JanusAI V2.
# It handles configuration loading, environment setup, agent creation,
# and the main training loop. It also supports command-line arguments for
# flexibility in training parameters and environment settings.
# The script is designed to be modular and extensible, allowing for easy
# integration of new features or changes to the training process.
