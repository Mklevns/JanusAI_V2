# janus/agents/components/vae.py
"""
Variational Autoencoder (VAE) components for the World Model.

This module implements the Vision (V) component that learns to encode
high-dimensional observations into compact latent representations. It provides
a base VAE class and specialized implementations for vector and image data.
"""

import random
from dataclasses import dataclass, asdict, field
from typing import Tuple, Optional, List, Dict, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def set_global_seed(seed: int):
    """
    Set global random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class VAEConfig:
    """
    Configuration dataclass for VAE architectures.
    """
    latent_dim: int = 32
    beta: float = 1.0  # Weight for KL divergence (from beta-VAE)
    seed: Optional[int] = 42

    def to_dict(self):
        return asdict(self)


@dataclass
class VectorVAEConfig(VAEConfig):
    """Configuration for a standard VAE processing vector data."""
    input_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])


@dataclass
class ConvVAEConfig(VAEConfig):
    """Configuration for a Convolutional VAE processing image data."""
    input_channels: int = 3
    image_hw: int = 64
    hidden_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])


class BaseVAE(nn.Module):
    """
    Abstract base class for Variational Autoencoders.
    """
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.seed is not None:
            set_global_seed(config.seed)
        self.to(self.device)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Applies the reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # In inference, use the deterministic mean

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass."""
        x = x.to(self.device)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

    def loss(self, x: torch.Tensor, recon_x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the VAE loss (reconstruction + KL divergence)."""
        # Note: using 'sum' and then dividing by batch size is more stable
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        total_loss = recon_loss + self.config.beta * kl_loss
        return total_loss, {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
        }

    def get_latent(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Gets the latent representation for a given input."""
        with torch.no_grad():
            x = x.to(self.device)
            mu, logvar = self.encode(x)
            return mu if deterministic else self.reparameterize(mu, logvar)

    @staticmethod
    def preprocess_observation(obs: np.ndarray, target_size: Tuple[int, int] = (64, 64)) -> torch.Tensor:
        """Preprocesses an observation image for model input."""
        if not isinstance(obs, torch.Tensor):
             obs = torch.from_numpy(obs).float()

        if obs.dim() == 3 and obs.shape[-1] in [1, 3]: # HWC to CHW
            obs = obs.permute(2, 0, 1)
        if obs.dim() == 2: # HW -> 1HW
            obs = obs.unsqueeze(0)
        if obs.dim() == 3: # CHW -> 1CHW
            obs = obs.unsqueeze(0)

        if obs.shape[-2:] != target_size:
            obs = F.interpolate(obs, size=target_size, mode="bilinear", align_corners=False)

        if obs.max() > 1.0: # Normalize to [0, 1]
            obs = obs / 255.0
        return obs


class ConvVAE(BaseVAE):
    """Convolutional VAE for image-based observations."""
    def __init__(self, config: ConvVAEConfig):
        super().__init__(config)
        self.config: ConvVAEConfig # For type hinting

        self.encoder = self._build_encoder()
        self._calculate_conv_output_size()

        self.fc_mu = nn.Linear(self.conv_output_size, config.latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, config.latent_dim)
        self.fc_decode = nn.Linear(config.latent_dim, self.conv_output_size)
        self.decoder = self._build_decoder()

    def _build_encoder(self) -> nn.Sequential:
        layers = []
        in_channels = self.config.input_channels
        for out_channels in self.config.hidden_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _calculate_conv_output_size(self):
        dummy_input = torch.zeros(1, self.config.input_channels, self.config.image_hw, self.config.image_hw)
        with torch.no_grad():
            output = self.encoder(dummy_input)
            self.conv_output_shape = output.shape[1:]
            self.conv_output_size = output.view(1, -1).size(1)

    def _build_decoder(self) -> nn.Sequential:
        layers = []
        hidden_channels = list(reversed(self.config.hidden_channels))
        for i, in_channels in enumerate(hidden_channels):
            out_channels = (
                hidden_channels[i + 1] if i + 1 < len(hidden_channels)
                else self.config.input_channels
            )
            is_last = i == len(hidden_channels) - 1
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels) if not is_last else nn.Identity(),
                nn.ReLU(inplace=True) if not is_last else nn.Sigmoid(),
            ])
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(h.size(0), *self.conv_output_shape)
        return self.decoder(h)

# --- Demonstration ---

def run_conv_vae_demo():
    """Demonstrates the Convolutional VAE with image data."""
    print("--- Testing Convolutional VAE ---")
    config = ConvVAEConfig(
        input_channels=3,
        image_hw=64,
        latent_dim=64,
        beta=0.5
    )
    print("Config:", config.to_dict())
    conv_vae = ConvVAE(config).to("cuda" if torch.cuda.is_available() else "cpu")

    # Test preprocessing
    mock_image_np = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    preprocessed = ConvVAE.preprocess_observation(mock_image_np)
    print(f"\nPreprocessing successful. Input shape: {mock_image_np.shape}, Output shape: {preprocessed.shape}")
    assert preprocessed.shape == (1, 3, 64, 64)

    # Test forward and backward pass
    mock_batch = torch.rand(16, 3, 64, 64)
    recon_images, mu, logvar, z = conv_vae(mock_batch)
    print(f"Forward pass successful. Reconstruction shape: {recon_images.shape}")
    assert recon_images.shape == mock_batch.shape

    loss, loss_dict = conv_vae.loss(mock_batch, recon_images, mu, logvar)
    loss.backward()
    print("Loss calculation and backward pass successful.")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    # Test latent vector retrieval
    latent_z = conv_vae.get_latent(preprocessed)
    print(f"Latent vector retrieval successful. Shape: {latent_z.shape}\n")
    assert latent_z.shape == (1, config.latent_dim)


if __name__ == '__main__':
    run_conv_vae_demo()
    print("All VAE tests passed! ✓")
