"""
Production-ready training script for pre-training World Model components (VAE and MDN-RNN).
Includes gradient clipping, RNG seeding, tensorboard cleanup, and data preprocessing improvements.
"""
import logging
import time
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import yaml

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, asdict
import torchvision.transforms as T
from tqdm import tqdm

# Assuming these are the correct import paths
from janus.agents.components.vae import VariationalAutoencoder, VAEConfig
from janus.agents.components.mdn_rnn import MDN_RNN, MDNRNNConfig, MDNRNNTrainer
from janus.utils.data import ExperienceDataset, preprocess_observation, get_env_info
from janus.utils.logging import setup_logging

# It's better to have a centralized logging setup
setup_logging()
logger = logging.getLogger(__name__)

# It's good practice to have transformations as reusable components
DATA_RESIZE = T.Compose([
    T.ToTensor(),
    T.Resize((64, 64), antialias=True) # antialias=True is important for quality
])

@dataclass
class WorldModelTrainingConfig:
    """Configuration for training the World Model."""
    # Data collection
    num_episodes: int = 1000
    max_episode_length: int = 500
    data_dir: str = "data/world_model"
    # VAE Training
    vae_epochs: int = 20
    vae_batch_size: int = 64
    vae_learning_rate: float = 1e-4
    vae_latent_dim: int = 32
    # MDN-RNN Training
    mdn_epochs: int = 30
    mdn_batch_size: int = 32
    mdn_sequence_length: int = 50
    mdn_learning_rate: float = 1e-3
    mdn_hidden_size: int = 256
    mdn_num_mixtures: int = 5
    # General
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_interval: int = 5
    log_interval: int = 20
    checkpoint_dir: str = "checkpoints/world_model"
    seed: int = 42

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f)

    @classmethod
    def load(cls, path: Path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class WorldModelTrainer:
    """Orchestrates the training of the VAE and MDN-RNN."""

    def __init__(self, env, config: WorldModelTrainingConfig):
        self.env = env
        self.config = config
        self.device = torch.device(config.device)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.data_dir = Path(config.data_dir)

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.checkpoint_dir / "logs")

        # Get environment information
        obs_shape, action_dim, self.is_continuous = get_env_info(self.env)
        self.observation_shape = (obs_shape[-1], obs_shape[0], obs_shape[1]) if obs_shape[-1] in [1, 3] else obs_shape
        self.action_dim = action_dim

        # Set seed for reproducibility
        self.set_seed(self.config.seed)

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def collect_random_data(self) -> Path:
        """Collects random rollouts from the environment and saves them."""
        data_path = self.data_dir / "random_rollouts.npz"
        if data_path.exists():
            logger.info(f"Loading existing random data from {data_path}")
            return data_path

        logger.info(f"Collecting {self.config.num_episodes} episodes of random data...")
        all_obs, all_actions = [], []

        for _ in tqdm(range(self.config.num_episodes), desc="Collecting Data"):
            obs, _ = self.env.reset()
            episode_obs, episode_actions = [preprocess_observation(obs, DATA_RESIZE)], []

            for _ in range(self.config.max_episode_length):
                action = self.env.action_space.sample()

                # Store the action
                if self.is_continuous:
                    episode_actions.append(action)
                    step_action = action
                else:
                    # The model will use one-hot, but we step with the integer action
                    one_hot_action = np.zeros(self.action_dim)
                    one_hot_action[action] = 1
                    episode_actions.append(one_hot_action)
                    step_action = action

                obs, _, done, truncated, _ = self.env.step(step_action)
                episode_obs.append(preprocess_observation(obs, DATA_RESIZE))

                if done or truncated:
                    break

            all_obs.append(np.array(episode_obs))
            all_actions.append(np.array(episode_actions))

        np.savez_compressed(data_path, observations=all_obs, actions=all_actions)
        logger.info(f"Saved random exploration data to {data_path}")
        return data_path

    def train_vae(self, data_path: Path) -> VariationalAutoencoder:
        """Trains the Variational Autoencoder."""
        logger.info("--- Starting VAE Training ---")
        vae_config = VAEConfig(
            input_channels=self.observation_shape[0],
            latent_dim=self.config.vae_latent_dim
        )
        vae = VariationalAutoencoder(vae_config).to(self.device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=self.config.vae_learning_rate)

        # Load all observations into a single array for VAE training
        data = np.load(data_path, allow_pickle=True)
        all_obs = np.concatenate(data['observations'], axis=0)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(all_obs).float())
        dataloader = DataLoader(dataset, batch_size=self.config.vae_batch_size, shuffle=True, num_workers=4, pin_memory=True)

        global_step = 0
        for epoch in range(self.config.vae_epochs):
            epoch_pbar = tqdm(dataloader, desc=f"VAE Epoch {epoch+1}/{self.config.vae_epochs}")
            for (obs_batch,) in epoch_pbar:
                obs_batch = obs_batch.to(self.device)
                recon, mu, logvar = vae(obs_batch)
                loss, loss_dict = vae.loss_function(recon, obs_batch, mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % self.config.log_interval == 0:
                    for key, value in loss_dict.items():
                        self.writer.add_scalar(f'VAE/{key}', value.item(), global_step)
                    epoch_pbar.set_postfix({k: v.item() for k, v in loss_dict.items()})
                global_step += 1

            if (epoch + 1) % self.config.save_interval == 0:
                torch.save(vae.state_dict(), self.checkpoint_dir / f'vae_epoch_{epoch+1}.pt')

        torch.save(vae.state_dict(), self.checkpoint_dir / 'vae_final.pt')
        logger.info(f"VAE training complete. Final model saved to {self.checkpoint_dir / 'vae_final.pt'}")
        return vae

    def train_mdn_rnn(self, vae: VariationalAutoencoder, data_path: Path) -> MDN_RNN:
        """Trains the Mixture Density Network RNN."""
        logger.info("--- Starting MDN-RNN Training ---")
        mdn_config = MDNRNNConfig(
            latent_size=self.config.vae_latent_dim,
            action_size=self.action_dim,
            hidden_size=self.config.mdn_hidden_size,
            num_mixtures=self.config.mdn_num_mixtures
        )
        mdn_rnn = MDN_RNN(mdn_config).to(self.device)
        trainer = MDNRNNTrainer(mdn_rnn, self.device, lr=self.config.mdn_learning_rate)

        vae.eval()

        # Create dataset of latent sequences
        data = np.load(data_path, allow_pickle=True)
        dataset = ExperienceDataset(
            observations=data['observations'],
            actions=data['actions'],
            sequence_length=self.config.mdn_sequence_length,
            vae=vae,
            device=self.device
        )
        dataloader = DataLoader(dataset, batch_size=self.config.mdn_batch_size, shuffle=True, num_workers=4, pin_memory=True)

        global_step = 0
        for epoch in range(self.config.mdn_epochs):
            epoch_pbar = tqdm(dataloader, desc=f"MDN Epoch {epoch+1}/{self.config.mdn_epochs}")
            for batch in epoch_pbar:
                latents = batch['latents'].to(self.device)
                actions = batch['actions'].to(self.device)

                if latents.size(1) < 2: continue

                loss, grad_norm = trainer.train_step(latents, actions, clip_grad_norm=1.0)

                if global_step % self.config.log_interval == 0:
                    self.writer.add_scalar('MDN/loss', loss, global_step)
                    self.writer.add_scalar('MDN/grad_norm', grad_norm, global_step)
                    epoch_pbar.set_postfix({'loss': loss, 'grad_norm': grad_norm})
                global_step += 1

            if (epoch + 1) % self.config.save_interval == 0:
                torch.save(mdn_rnn.state_dict(), self.checkpoint_dir / f'mdn_rnn_epoch_{epoch+1}.pt')

        torch.save(mdn_rnn.state_dict(), self.checkpoint_dir / 'mdn_rnn_final.pt')
        logger.info(f"MDN-RNN training complete. Final model saved to {self.checkpoint_dir / 'mdn_rnn_final.pt'}")
        return mdn_rnn

    def run(self):
        """Runs the full training pipeline."""
        # 1. Collect Data
        data_path = self.collect_random_data()

        # 2. Train VAE
        vae = self.train_vae(data_path)

        # 3. Train MDN-RNN
        self.train_mdn_rnn(vae, data_path)

        self.writer.close()
        logger.info("World Model training finished.")
