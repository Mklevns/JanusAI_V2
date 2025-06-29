# janus/agents/world_model_agent.py
"""
World Model Agent that integrates VAE and MDN-RNN with the PPO Controller.
"""

import logging
import random
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
from torch import nn

from janus.agents.ppo_agent import PPOAgent, NetworkConfig
from janus.agents.components.vae import VariationalAutoencoder, VAEConfig
from janus.agents.components.mdn_rnn import MDN_RNN, MDNRNNConfig

# Set reproducibility seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

logger = logging.getLogger(__name__)

class WorldModelAgent(nn.Module):
    """
    An agent that combines a World Model (VAE + MDN-RNN) with a PPO controller.

    This class manages the full lifecycle of the model-based RL agent, including encoding
    observations, generating actions, simulating latent trajectories, and saving/loading components.
    """
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        vae_config: VAEConfig,
        mdn_config: MDNRNNConfig,
        controller_config: NetworkConfig,
        device: str = "cpu",
    ):
        """
        Initialize the WorldModelAgent.

        Args:
            observation_shape: Shape of the input observation.
            action_dim: Dimension of the action space.
            vae_config: Configuration for the VAE component.
            mdn_config: Configuration for the MDN-RNN component.
            controller_config: Configuration for the PPO controller.
            device: The device to run computations on ('cpu' or 'cuda').
        """
        super().__init__()
        self.device = torch.device(device)
        self.vae = VariationalAutoencoder(vae_config).to(self.device)
        self.mdn_rnn = MDN_RNN(mdn_config).to(self.device)
        self.controller = PPOAgent(
            input_dim=vae_config.latent_dim + mdn_config.hidden_dim,
            output_dim=action_dim,
            config=controller_config
        ).to(self.device)

        self.current_latent: Optional[torch.Tensor] = None
        self.current_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        logger.info(
            "WorldModelAgent initialized with latent_dim=%d, hidden_dim=%d, action_dim=%d",
            vae_config.latent_dim, mdn_config.hidden_dim, action_dim
        )

    def reset(self):
        """Reset internal latent and hidden states."""
        self.current_latent = None
        self.current_hidden = None

    def encode_observation(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Encode an observation into a latent vector using the VAE.

        Args:
            observation: Input observation tensor.

        Returns:
            Latent representation tensor.
        """
        observation = observation.to(self.device)
        return self.vae.encode(observation)

    def act(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Generate an action given the current observation.

        Args:
            observation: Input observation tensor.
            deterministic: Whether to sample deterministically.
            return_components: Whether to return intermediate representations.

        Returns:
            Tuple of (action, log probability, optional dict of {'z', 'h'}).
        """
        z = self.encode_observation(observation)

        if self.current_hidden is None:
            batch_size = z.size(0)
            self.current_hidden = self.mdn_rnn.init_hidden(batch_size, self.device)

        if isinstance(self.current_hidden, tuple):
            h = self.current_hidden[0].squeeze(0)
        else:
            h = self.current_hidden.squeeze(0)

        controller_input = torch.cat([z, h], dim=-1)
        action, log_prob, _ = self.controller.act(controller_input, deterministic=deterministic)

        # Update hidden state for next time step
        _, _, _, self.current_hidden = self.mdn_rnn(
            z.unsqueeze(1), action.unsqueeze(1), self.current_hidden
        )

        if return_components:
            return action, log_prob, {"z": z, "h": h}
        return action, log_prob, None

    def imagine_trajectory(
        self,
        initial_obs: torch.Tensor,
        horizon: int = 10
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
        """
        Simulate a latent trajectory starting from an initial observation.

        Args:
            initial_obs: Initial observation tensor.
            horizon: Number of steps to imagine.

        Returns:
            Tuple of (latents, actions, rewards) over the imagined trajectory.
        """
        z = self.encode_observation(initial_obs)

        if self.current_hidden is None:
            batch_size = z.size(0)
            self.current_hidden = self.mdn_rnn.init_hidden(batch_size, self.device)

        latents = [z]
        actions: List[torch.Tensor] = []
        rewards: List[float] = []  # Placeholder - would need a reward model

        for _ in range(horizon):
            if self.current_hidden is None:
                raise RuntimeError("Hidden state is None during trajectory imagination.")

            if isinstance(self.current_hidden, tuple):
                h = self.current_hidden[0].squeeze(0)
            else:
                h = self.current_hidden.squeeze(0)

            controller_input = torch.cat([z, h], dim=-1)
            action, _, _ = self.controller.act(controller_input)
            actions.append(action)

            pi, mu, sigma, self.current_hidden = self.mdn_rnn(
                z.unsqueeze(1), action.unsqueeze(1), self.current_hidden
            )
            z = self.mdn_rnn.sample(pi[:, 0], mu[:, 0], sigma[:, 0])
            latents.append(z)

        return latents, actions, rewards

    def save_components(self, save_dir: Path) -> None:
        """
        Save model and config components to the specified directory.

        Args:
            save_dir: Path to the directory where components will be saved.
        """
        torch.save({
            'vae': self.vae.state_dict(),
            'mdn_rnn': self.mdn_rnn.state_dict(),
            'controller': self.controller.state_dict()
        }, save_dir / 'world_model.pt')

        torch.save({
            'vae_config': self.vae.config,
            'mdn_config': self.mdn_rnn.config,
            'controller_config': self.controller.config
        }, save_dir / 'world_model_config.pt')

        logger.info("World model components saved to %s", save_dir)

    def load_components(self, load_dir: Path) -> None:
        """
        Load model and config components from the specified directory.

        Args:
            load_dir: Path to the directory from which components will be loaded.
        """
        state_dict = torch.load(load_dir / 'world_model.pt', map_location=self.device)
        self.vae.load_state_dict(state_dict['vae'])
        self.mdn_rnn.load_state_dict(state_dict['mdn_rnn'])
        self.controller.load_state_dict(state_dict['controller'])

        self.controller.load_checkpoint(str(load_dir / 'controller.pt'))

        logger.info("World model components loaded from %s", load_dir)
