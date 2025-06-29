# File: janus/scripts/examples/train_world_model_ppo.py
"""
Train a PPO agent using a world model that combines a VAE and MDN-RNN.

This script demonstrates how to set up and train a WorldModelAgent, showcasing
the integration of custom agent components with the PPO trainer. It follows
the project's best practices by separating concerns into modular functions and
using configuration files for hyperparameters.
"""

# Standard library imports
import logging
import random  # Note: Used for reproducibility, not for cryptographic purposes
from pathlib import Path
from typing import List

# Third-party imports
import gymnasium as gym
import numpy as np
import torch
import yaml

# Local application imports
from janus.agents.components.mdn_rnn import MDNRNNConfig
from janus.agents.components.vae import VAEConfig
from janus.agents.world_model_agent import WorldModelAgent
from janus.training.ppo.config import PPOConfig
# Assuming WorldModelPPOTrainer exists and is the correct trainer for this agent
from janus.training.ppo.trainer import WorldModelPPOTrainer

# Configure logger for the script
logger = logging.getLogger(__name__)


def setup_environment(env_name: str, num_envs: int, seed: int) -> List[gym.Env]:
    """
    Initializes and seeds the training environments.

    Args:
        env_name: The name of the gymnasium environment.
        num_envs: The number of parallel environments to create.
        seed: The random seed.

    Returns:
        A list of initialized gymnasium environments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    envs = [gym.make(env_name) for _ in range(num_envs)]
    # The following loop iterates over a list of environment objects, not a NumPy array.
    for env in envs:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    logger.info("Created %d instances of '%s' environment.", num_envs, env_name)
    return envs


def setup_agent(
    envs: List[gym.Env],
    vae_config: VAEConfig,
    mdn_config: MDNRNNConfig,
    checkpoint_dir: Path,
    device: str,
) -> WorldModelAgent:
    """
    Initializes the WorldModelAgent and loads any pretrained components.

    Args:
        envs: A list of environments to infer shapes and action spaces.
        vae_config: Configuration for the VAE component.
        mdn_config: Configuration for the MDN-RNN component.
        checkpoint_dir: Directory to check for pretrained model weights.
        device: The device to run the agent on ('cpu' or 'cuda').

    Returns:
        An initialized WorldModelAgent instance.
    """
    obs, _ = envs[0].reset()
    obs_shape = obs.shape if obs.ndim == 3 else (1,) + obs.shape
    action_dim = (
        envs[0].action_space.n
        if hasattr(envs[0].action_space, "n")
        else envs[0].action_space.shape[0]
    )
    continuous = not hasattr(envs[0].action_space, "n")

    agent = WorldModelAgent(
        observation_shape=obs_shape,
        action_dim=action_dim,
        vae_config=vae_config,
        mdn_config=mdn_config,
        continuous_actions=continuous,
        device=device,
    )

    # Load pretrained components if they exist
    if (checkpoint_dir / "vae_final.pt").exists():
        agent.vae.load_state_dict(torch.load(checkpoint_dir / "vae_final.pt"))
        logger.info("Loaded pre-trained VAE from checkpoint.")
    if (checkpoint_dir / "mdn_final.pt").exists():
        agent.mdn_rnn.load_state_dict(torch.load(checkpoint_dir / "mdn_final.pt"))
        logger.info("Loaded pre-trained MDN-RNN from checkpoint.")

    return agent


def train_agent(
    agent: WorldModelAgent,
    envs: List[gym.Env],
    ppo_config: PPOConfig,
    trainer_settings: dict,
    total_timesteps: int,
):
    """
    Initializes and runs the PPO trainer.

    Args:
        agent: The agent to be trained.
        envs: The list of environments for training.
        ppo_config: The configuration for the PPO algorithm.
        trainer_settings: Additional settings for the trainer.
        total_timesteps: The total number of timesteps for training.
    """
    trainer = WorldModelPPOTrainer(
        world_model_agent=agent,
        envs=envs,
        config=ppo_config,
        **trainer_settings,
    )

    logger.info("Starting World Model PPO training...")
    trainer.train(total_timesteps=total_timesteps, rollout_length=2048)


def main(
    env_name: str,
    total_timesteps: int,
    num_envs: int,
    seed: int,
    config_path: str,
):
    """
    Main function to orchestrate the training of the World Model PPO agent.

    Args:
        env_name: Name of the gymnasium environment.
        total_timesteps: Total number of training steps.
        num_envs: Number of parallel environments.
        seed: Random seed for reproducibility.
        config_path: Path to the YAML configuration file for the agent and trainer.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)

    vae_config = VAEConfig(**full_config['vae_config'])
    mdn_config = MDNRNNConfig(**full_config['mdn_config'])
    ppo_config = PPOConfig(**full_config['ppo_config'])
    trainer_settings = full_config['trainer_settings']

    checkpoint_dir = Path("checkpoints/world_model")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    envs = setup_environment(env_name, num_envs, seed)
    agent = setup_agent(envs, vae_config, mdn_config, checkpoint_dir, device)
    train_agent(agent, envs, ppo_config, trainer_settings, total_timesteps)

    final_checkpoint_dir = Path("checkpoints/world_model_ppo_final")
    agent.save_components(final_checkpoint_dir)
    logger.info("Training completed! Final components saved to %s", final_checkpoint_dir)

    for env in envs:
        env.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Example of how to run the training.
    # A dedicated YAML config file should be created for this.
    main(
        env_name="LunarLander-v2",
        total_timesteps=500_000,
        num_envs=4,
        seed=42,
        config_path="configs/lunar_lander_wm_ppo.yaml",  # Assumes this file exists
    )
