import random
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def set_global_seed(seed: int):
    """
    Set global random seeds for reproducibility.

    Args:
        seed (int): Random seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class VAEConfig:
    """
    Configuration dataclass for the VAE architecture.
    """
    input_channels: int = 3
    input_height: int = 64
    input_width: int = 64
    latent_dim: int = 32
    hidden_channels: Optional[List[int]] = None
    beta: float = 1.0
    seed: Optional[int] = 42

    def __post_init__(self):
        """Set default hidden channels if none are provided."""
        if self.hidden_channels is None:
            self.hidden_channels = [32, 64, 128, 256]

    def to_dict(self):
        """Convert the config to a dictionary."""
        return asdict(self)


class VariationalAutoencoder(nn.Module):
    """
    Convolutional Variational Autoencoder for image-based inputs.
    """
    def __init__(self, config: VAEConfig):
        """
        Initialize the VAE architecture based on provided config.

        Args:
            config (VAEConfig): Configuration object.
        """
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.seed is not None:
            set_global_seed(config.seed)

        self._calculate_conv_output_size()
        self.encoder = self._build_encoder()
        self.fc_mu = nn.Linear(self.conv_output_size, config.latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, config.latent_dim)
        self.fc_decode = nn.Linear(config.latent_dim, self.conv_output_size)
        self.decoder = self._build_decoder()

        self.to(self.device)

    def _calculate_conv_output_size(self):
        """
        Calculate the output size after the encoder convolutions.
        """
        dummy_input = torch.zeros(
            1,
            self.config.input_channels,
            self.config.input_height,
            self.config.input_width,
        )
        temp_encoder = self._build_encoder()
        with torch.no_grad():
            output = temp_encoder(dummy_input)
            self.conv_output_size = output.view(1, -1).size(1)
            self.conv_output_shape = output.shape[1:]

    def _build_encoder(self) -> nn.Sequential:
        """
        Construct the encoder CNN layers.

        Returns:
            nn.Sequential: Encoder network.
        """
        layers = []
        in_channels = self.config.input_channels
        for out_channels in self.config.hidden_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Sequential:
        """
        Construct the decoder CNN layers.

        Returns:
            nn.Sequential: Decoder network.
        """
        layers = []
        hidden_channels = list(reversed(self.config.hidden_channels))
        for i, in_channels in enumerate(hidden_channels):
            out_channels = (
                hidden_channels[i + 1] if i + 1 < len(hidden_channels)
                else self.config.input_channels
            )
            is_last = i == len(hidden_channels) - 1
            layers.extend([
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=4, stride=2, padding=1
                ),
                nn.InstanceNorm2d(out_channels) if not is_last else nn.Identity(),
                nn.ReLU(inplace=True) if not is_last else nn.Sigmoid(),
            ])
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input images into latent space parameters.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log variance.
        """
        x = x.to(self.device)
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Apply the reparameterization trick.

        Args:
            mu (torch.Tensor): Mean.
            logvar (torch.Tensor): Log variance.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstructed image.

        Args:
            z (torch.Tensor): Latent vector.

        Returns:
            torch.Tensor: Reconstructed image.
        """
        z = z.to(self.device)
        h = self.fc_decode(z)
        h = h.view(h.size(0), *self.conv_output_shape)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass: encode, reparameterize, decode.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Reconstruction, mean, and log variance.
        """
        x = x.to(self.device)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def loss(self, x: torch.Tensor, reconstruction: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute VAE loss combining reconstruction and KL divergence.

        Args:
            x (torch.Tensor): Original input.
            reconstruction (torch.Tensor): Reconstructed image.
            mu (torch.Tensor): Mean from encoder.
            logvar (torch.Tensor): Log variance from encoder.

        Returns:
            Tuple[torch.Tensor, dict]: Total loss and component breakdown.
        """
        recon_loss = F.mse_loss(reconstruction, x, reduction="sum") / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        total_loss = recon_loss + self.config.beta * kl_loss
        return total_loss, {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item()
        }

    def get_latent(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get latent vector from input.

        Args:
            x (torch.Tensor): Input image.
            deterministic (bool): Use mean if True, else sample.

        Returns:
            torch.Tensor: Latent representation.
        """
        with torch.no_grad():
            mu, logvar = self.encode(x)
            return mu if deterministic else self.reparameterize(mu, logvar)


def preprocess_observation(obs: np.ndarray, target_size: Tuple[int, int] = (64, 64)) -> torch.Tensor:
    """
    Preprocess an observation for model input.

    Args:
        obs (np.ndarray): Image array.
        target_size (Tuple[int, int]): Resize target.

    Returns:
        torch.Tensor: Preprocessed tensor.
    """
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).float()
    if obs.dim() == 2:
        obs = obs.unsqueeze(0).unsqueeze(0)
    elif obs.dim() == 3:
        if obs.shape[-1] in [1, 3]:
            obs = obs.permute(2, 0, 1)
        obs = obs.unsqueeze(0)
    if obs.shape[-2:] != target_size:
        obs = F.interpolate(obs, size=target_size, mode="bilinear", align_corners=False)
    if obs.max() > 1.0:
        obs = obs / 255.0
    return obs
