# janus/agents/world_model/components.py
"""
World Model components based on "World Models" paper by Ha & Schmidhuber.
Implements VAE for visual encoding and MDN-RNN for temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MixtureSameFamily
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class WorldModelConfig:
    """Configuration for World Model components."""
    # VAE parameters
    vae_latent_dim: int = 32
    vae_encoder_layers: list = None  # e.g., [64, 128, 256]
    vae_decoder_layers: list = None  # e.g., [256, 128, 64]
    vae_beta: float = 1.0  # KL weight in VAE loss

    # MDN-RNN parameters
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1
    rnn_mixture_components: int = 5  # Number of Gaussian mixtures
    rnn_temperature: float = 1.0

    # Training parameters
    learning_rate: float = 1e-4
    sequence_length: int = 100
    batch_size: int = 32

    def __post_init__(self):
        if self.vae_encoder_layers is None:
            self.vae_encoder_layers = [64, 128, 256]
        if self.vae_decoder_layers is None:
            self.vae_decoder_layers = [256, 128, 64]


class VariationalAutoencoder(nn.Module):
    """
    VAE for encoding high-dimensional observations into compact latent vectors.
    The 'V' component of the World Model.
    """

    def __init__(self, input_dim: int, config: WorldModelConfig):
        """
        Args:
            input_dim: Dimension of input observations
            config: World model configuration
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = config.vae_latent_dim
        self.beta = config.vae_beta

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in config.vae_encoder_layers:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent representation
        self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, self.latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = self.latent_dim
        for hidden_dim in config.vae_decoder_layers:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input observations [batch_size, input_dim]

        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            log_var: Log variance of latent distribution [batch_size, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.

        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution

        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction.

        Args:
            z: Latent vector [batch_size, latent_dim]

        Returns:
            Reconstructed observation [batch_size, input_dim]
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through VAE.

        Args:
            x: Input observations

        Returns:
            recon_x: Reconstructed observations
            mu: Latent mean
            log_var: Latent log variance
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

    def loss(self, x: torch.Tensor, recon_x: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss (reconstruction + KL divergence).

        Args:
            x: Original observations
            recon_x: Reconstructed observations
            mu: Latent mean
            log_var: Latent log variance

        Returns:
            Dictionary with total loss and components
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'kl_divergence': kl_loss
        }


class MixtureDensityRNN(nn.Module):
    """
    MDN-RNN for predicting future latent states.
    The 'M' (Memory) component of the World Model.
    """

    def __init__(self, latent_dim: int, action_dim: int, config: WorldModelConfig):
        """
        Args:
            latent_dim: Dimension of latent vectors from VAE
            action_dim: Dimension of action space
            config: World model configuration
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = config.rnn_hidden_dim
        self.num_layers = config.rnn_num_layers
        self.n_mixtures = config.rnn_mixture_components
        self.temperature = config.rnn_temperature

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=latent_dim + action_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )

        # MDN outputs
        # For each mixture component: weight, mean (latent_dim), variance (latent_dim)
        mdn_output_size = self.n_mixtures * (1 + 2 * latent_dim)
        self.mdn_head = nn.Linear(self.hidden_dim, mdn_output_size)

        # Optional: predict reward
        self.reward_head = nn.Linear(self.hidden_dim, 1)

        # Optional: predict done flag
        self.done_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, z: torch.Tensor, a: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Forward pass through MDN-RNN.

        Args:
            z: Latent vectors [batch_size, seq_len, latent_dim]
            a: Actions [batch_size, seq_len, action_dim]
            hidden: LSTM hidden state

        Returns:
            Dictionary containing predictions and hidden state
        """
        batch_size, seq_len = z.shape[:2]

        # Concatenate latent and action
        rnn_input = torch.cat([z, a], dim=-1)

        # LSTM forward pass
        lstm_out, hidden = self.lstm(rnn_input, hidden)

        # MDN predictions
        mdn_params = self.mdn_head(lstm_out)

        # Split MDN parameters
        stride = self.n_mixtures

        # Mixture weights (logits)
        pi_logits = mdn_params[..., :stride]  # [batch, seq, n_mixtures]

        # Means
        mu = mdn_params[..., stride:stride + stride * self.latent_dim]
        mu = mu.view(batch_size, seq_len, self.n_mixtures, self.latent_dim)

        # Log variances
        log_var = mdn_params[..., stride + stride * self.latent_dim:]
        log_var = log_var.view(batch_size, seq_len, self.n_mixtures, self.latent_dim)

        # Optional predictions
        reward_pred = self.reward_head(lstm_out).squeeze(-1)
        done_logits = self.done_head(lstm_out).squeeze(-1)

        return {
            'pi_logits': pi_logits,
            'mu': mu,
            'log_var': log_var,
            'reward_pred': reward_pred,
            'done_logits': done_logits,
            'hidden': hidden,
            'lstm_out': lstm_out
        }

    def sample_prediction(self, pi_logits: torch.Tensor, mu: torch.Tensor,
                         log_var: torch.Tensor) -> torch.Tensor:
        """
        Sample from the mixture of Gaussians.

        Args:
            pi_logits: Mixture weights logits [batch_size, n_mixtures]
            mu: Means [batch_size, n_mixtures, latent_dim]
            log_var: Log variances [batch_size, n_mixtures, latent_dim]

        Returns:
            Sampled latent vector [batch_size, latent_dim]
        """
        batch_size = pi_logits.shape[0]

        # Sample mixture component
        pi_logits = pi_logits / self.temperature
        mixture_dist = Categorical(logits=pi_logits)
        mixture_idx = mixture_dist.sample()  # [batch_size]

        # Get parameters for selected mixtures
        batch_idx = torch.arange(batch_size)
        selected_mu = mu[batch_idx, mixture_idx]  # [batch_size, latent_dim]
        selected_log_var = log_var[batch_idx, mixture_idx]

        # Sample from selected Gaussian
        std = torch.exp(0.5 * selected_log_var)
        eps = torch.randn_like(std)
        z_pred = selected_mu + eps * std

        return z_pred

    def loss(self, z_true: torch.Tensor, pi_logits: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor,
             reward_true: Optional[torch.Tensor] = None,
             reward_pred: Optional[torch.Tensor] = None,
             done_true: Optional[torch.Tensor] = None,
             done_logits: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute MDN-RNN loss.

        Args:
            z_true: True next latent vectors [batch_size, seq_len, latent_dim]
            pi_logits: Mixture weights logits
            mu: Mixture means
            log_var: Mixture log variances
            reward_true: True rewards (optional)
            reward_pred: Predicted rewards (optional)
            done_true: True done flags (optional)
            done_logits: Predicted done logits (optional)

        Returns:
            Dictionary with loss components
        """
        batch_size, seq_len = z_true.shape[:2]

        # Reshape for loss computation
        z_true_flat = z_true.view(-1, self.latent_dim)
        pi_logits_flat = pi_logits.view(-1, self.n_mixtures)
        mu_flat = mu.view(-1, self.n_mixtures, self.latent_dim)
        log_var_flat = log_var.view(-1, self.n_mixtures, self.latent_dim)

        # Create mixture distribution
        mix_dist = Categorical(logits=pi_logits_flat)

        # Create component distributions
        std = torch.exp(0.5 * log_var_flat)
        comp_dist = Normal(mu_flat, std)

        # Create mixture of Gaussians
        gmm = MixtureSameFamily(mix_dist, comp_dist)

        # Negative log likelihood
        nll = -gmm.log_prob(z_true_flat.unsqueeze(1)).mean()

        losses = {'nll': nll}

        # Optional: reward prediction loss
        if reward_true is not None and reward_pred is not None:
            reward_loss = F.mse_loss(reward_pred.view(-1), reward_true.view(-1))
            losses['reward'] = reward_loss

        # Optional: done prediction loss
        if done_true is not None and done_logits is not None:
            done_loss = F.binary_cross_entropy_with_logits(
                done_logits.view(-1), done_true.view(-1).float()
            )
            losses['done'] = done_loss

        # Total loss
        losses['total'] = sum(losses.values())

        return losses


class WorldModel(nn.Module):
    """
    Complete World Model combining VAE and MDN-RNN.
    """

    def __init__(self, observation_dim: int, action_dim: int, config: WorldModelConfig):
        """
        Args:
            observation_dim: Dimension of observations
            action_dim: Dimension of actions
            config: World model configuration
        """
        super().__init__()
        self.config = config

        # Vision model (VAE)
        self.vae = VariationalAutoencoder(observation_dim, config)

        # Memory model (MDN-RNN)
        self.mdn_rnn = MixtureDensityRNN(config.vae_latent_dim, action_dim, config)

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observation to latent vector.

        Args:
            obs: Observation [batch_size, obs_dim]

        Returns:
            Latent vector [batch_size, latent_dim]
        """
        with torch.no_grad():
            mu, log_var = self.vae.encode(obs)
            z = self.vae.reparameterize(mu, log_var)
        return z

    def predict_next_latent(self, z: torch.Tensor, a: torch.Tensor,
                           hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Predict next latent state given current latent and action.

        Args:
            z: Current latent state [batch_size, latent_dim]
            a: Action [batch_size, action_dim]
            hidden: RNN hidden state

        Returns:
            Dictionary with predictions
        """
        # Add sequence dimension
        z = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
        a = a.unsqueeze(1)  # [batch_size, 1, action_dim]

        # Get predictions
        outputs = self.mdn_rnn(z, a, hidden)

        # Sample next latent
        pi_logits = outputs['pi_logits'].squeeze(1)
        mu = outputs['mu'].squeeze(1)
        log_var = outputs['log_var'].squeeze(1)

        z_next = self.mdn_rnn.sample_prediction(pi_logits, mu, log_var)

        return {
            'z_next': z_next,
            'reward_pred': outputs['reward_pred'].squeeze(1),
            'done_pred': torch.sigmoid(outputs['done_logits'].squeeze(1)),
            'hidden': outputs['hidden']
        }

    def imagine_trajectory(self, initial_obs: torch.Tensor,
                          action_sequence: torch.Tensor,
                          horizon: int) -> Dict[str, torch.Tensor]:
        """
        Imagine a trajectory by rolling out the world model.

        Args:
            initial_obs: Initial observation [batch_size, obs_dim]
            action_sequence: Planned actions [batch_size, horizon, action_dim]
            horizon: Number of steps to imagine

        Returns:
            Dictionary with imagined trajectory
        """
        batch_size = initial_obs.shape[0]

        # Encode initial observation
        z = self.encode_observation(initial_obs)

        # Storage for trajectory
        latents = [z]
        rewards = []
        dones = []

        # Hidden state
        hidden = None

        # Roll out trajectory
        for t in range(horizon):
            # Get action for this timestep
            a = action_sequence[:, t]

            # Predict next state
            outputs = self.predict_next_latent(z, a, hidden)

            # Store predictions
            z = outputs['z_next']
            latents.append(z)
            rewards.append(outputs['reward_pred'])
            dones.append(outputs['done_pred'])
            hidden = outputs['hidden']

        # Stack results
        latents = torch.stack(latents, dim=1)  # [batch_size, horizon+1, latent_dim]
        rewards = torch.stack(rewards, dim=1)  # [batch_size, horizon]
        dones = torch.stack(dones, dim=1)      # [batch_size, horizon]

        return {
            'latents': latents,
            'rewards': rewards,
            'dones': dones
        }


# Demonstration
if __name__ == '__main__':
    """Demonstrate World Model components."""
    import matplotlib.pyplot as plt

    print("=== World Model Components Demo ===\n")

    # Configuration
    config = WorldModelConfig(
        vae_latent_dim=32,
        vae_encoder_layers=[64, 128],
        vae_decoder_layers=[128, 64],
        rnn_hidden_dim=256,
        rnn_mixture_components=5
    )

    # Dimensions
    obs_dim = 100
    action_dim = 4
    batch_size = 16
    seq_len = 10

    # Create world model
    world_model = WorldModel(obs_dim, action_dim, config)
    print(f"Created World Model with {sum(p.numel() for p in world_model.parameters())} parameters")

    # Test VAE
    print("\n1. Testing VAE...")
    obs = torch.randn(batch_size, obs_dim)
    recon_obs, mu, log_var = world_model.vae(obs)
    vae_loss = world_model.vae.loss(obs, recon_obs, mu, log_var)
    print(f"   VAE loss: {vae_loss['total'].item():.4f}")
    print(f"   Reconstruction loss: {vae_loss['reconstruction'].item():.4f}")
    print(f"   KL divergence: {vae_loss['kl_divergence'].item():.4f}")

    # Test MDN-RNN
    print("\n2. Testing MDN-RNN...")
    z_seq = torch.randn(batch_size, seq_len, config.vae_latent_dim)
    a_seq = torch.randn(batch_size, seq_len, action_dim)

    outputs = world_model.mdn_rnn(z_seq, a_seq)
    print(f"   Pi logits shape: {outputs['pi_logits'].shape}")
    print(f"   Mu shape: {outputs['mu'].shape}")
    print(f"   Log var shape: {outputs['log_var'].shape}")

    # Test prediction sampling
    print("\n3. Testing prediction sampling...")
    pi_logits = outputs['pi_logits'][:, 0]  # First timestep
    mu = outputs['mu'][:, 0]
    log_var = outputs['log_var'][:, 0]

    z_pred = world_model.mdn_rnn.sample_prediction(pi_logits, mu, log_var)
    print(f"   Predicted latent shape: {z_pred.shape}")

    # Test trajectory imagination
    print("\n4. Testing trajectory imagination...")
    initial_obs = torch.randn(4, obs_dim)
    action_seq = torch.randn(4, 20, action_dim)

    trajectory = world_model.imagine_trajectory(initial_obs, action_seq, horizon=20)
    print(f"   Imagined latents shape: {trajectory['latents'].shape}")
    print(f"   Imagined rewards shape: {trajectory['rewards'].shape}")
    print(f"   Imagined dones shape: {trajectory['dones'].shape}")

    # Visualize imagined rewards
    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(trajectory['rewards'][i].detach().numpy(),
                label=f'Trajectory {i+1}', alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('Imagined Reward')
    plt.title('Imagined Reward Trajectories')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('imagined_rewards.png')
    print("\n   Saved visualization to 'imagined_rewards.png'")

    print("\n=== Demo completed successfully! ===")
