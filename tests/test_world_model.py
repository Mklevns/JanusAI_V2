# tests/test_world_model.py
"""
Unit tests for the World Model components (VAE, MDN-RNN, and WorldModelAgent).
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from janus.agents.components.vae import VariationalAutoencoder, VAEConfig, preprocess_observation
from janus.agents.components.mdn_rnn import MDN_RNN, MDNRNNConfig
from janus.agents.world_model_agent import WorldModelAgent


class TestVAE:
    """Test the Variational Autoencoder component."""

    @pytest.fixture
    def vae_config(self):
        """Create a test VAE configuration."""
        return VAEConfig(
            input_channels=3,
            input_height=64,
            input_width=64,
            latent_dim=16,
            hidden_channels=[16, 32],
            beta=1.0
        )

    @pytest.fixture
    def vae(self, vae_config):
        """Create a test VAE instance."""
        return VariationalAutoencoder(vae_config)

    def test_vae_initialization(self, vae, vae_config):
        """Test VAE initialization."""
        assert vae.config.latent_dim == 16
        assert vae.config.input_channels == 3
        assert vae.config.hidden_channels == [16, 32]

    def test_vae_forward(self, vae):
        """Test VAE forward pass."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 64, 64)

        reconstruction, mu, logvar = vae(x)

        assert reconstruction.shape == x.shape
        assert mu.shape == (batch_size, 16)
        assert logvar.shape == (batch_size, 16)

    def test_vae_loss(self, vae):
        """Test VAE loss calculation."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 64, 64)

        reconstruction, mu, logvar = vae(x)
        loss, loss_dict = vae.loss(x, reconstruction, mu, logvar)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert 'total_loss' in loss_dict
        assert 'recon_loss' in loss_dict
        assert 'kl_loss' in loss_dict

    def test_vae_encode_decode(self, vae):
        """Test encoding and decoding separately."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64)

        # Encode
        mu, logvar = vae.encode(x)
        assert mu.shape == (batch_size, 16)

        # Reparameterize
        z = vae.reparameterize(mu, logvar)
        assert z.shape == (batch_size, 16)

        # Decode
        reconstruction = vae.decode(z)
        assert reconstruction.shape == x.shape

    def test_get_latent(self, vae):
        """Test getting latent representations."""
        x = torch.randn(1, 3, 64, 64)

        # Stochastic
        z1 = vae.get_latent(x, deterministic=False)
        z2 = vae.get_latent(x, deterministic=False)
        assert not torch.allclose(z1, z2)  # Should be different due to sampling

        # Deterministic
        z3 = vae.get_latent(x, deterministic=True)
        z4 = vae.get_latent(x, deterministic=True)
        assert torch.allclose(z3, z4)  # Should be the same

    def test_preprocess_observation(self):
        """Test observation preprocessing utility."""
        # Test grayscale image
        obs_gray = np.random.rand(64, 64)
        processed = preprocess_observation(obs_gray)
        assert processed.shape == (1, 1, 64, 64)

        # Test RGB image (channel last)
        obs_rgb = np.random.rand(64, 64, 3)
        processed = preprocess_observation(obs_rgb)
        assert processed.shape == (1, 3, 64, 64)

        # Test resizing
        obs_large = np.random.rand(128, 128, 3)
        processed = preprocess_observation(obs_large, target_size=(64, 64))
        assert processed.shape == (1, 3, 64, 64)


class TestMDNRNN:
    """Test the MDN-RNN component."""

    @pytest.fixture
    def mdn_config(self):
        """Create a test MDN-RNN configuration."""
        return MDNRNNConfig(
            latent_dim=16,
            action_dim=4,
            hidden_dim=64,
            num_mixtures=3,
            rnn_layers=1,
            rnn_type="lstm"
        )

    @pytest.fixture
    def mdn_rnn(self, mdn_config):
        """Create a test MDN-RNN instance."""
        return MDN_RNN(mdn_config)

    def test_mdn_initialization(self, mdn_rnn, mdn_config):
        """Test MDN-RNN initialization."""
        assert mdn_rnn.config.latent_dim == 16
        assert mdn_rnn.config.action_dim == 4
        assert mdn_rnn.config.num_mixtures == 3

    def test_mdn_forward(self, mdn_rnn):
        """Test MDN-RNN forward pass."""
        batch_size = 2
        seq_len = 5

        z = torch.randn(batch_size, seq_len, 16)
        a = torch.randn(batch_size, seq_len, 4)
        hidden = mdn_rnn.init_hidden(batch_size, torch.device('cpu'))

        pi, mu, sigma, new_hidden = mdn_rnn(z, a, hidden)

        assert pi.shape == (batch_size, seq_len, 3)  # num_mixtures
        assert mu.shape == (batch_size, seq_len, 3, 16)  # num_mixtures x latent_dim
        assert sigma.shape == (batch_size, seq_len, 3, 16)
        assert isinstance(new_hidden, tuple)  # LSTM returns (h, c)

    def test_mdn_sample(self, mdn_rnn):
        """Test sampling from MDN output."""
        batch_size = 4

        pi = torch.softmax(torch.randn(batch_size, 3), dim=-1)
        mu = torch.randn(batch_size, 3, 16)
        sigma = torch.abs(torch.randn(batch_size, 3, 16)) + 0.1

        z_next = mdn_rnn.sample(pi, mu, sigma)
        assert z_next.shape == (batch_size, 16)

    def test_mdn_loss(self, mdn_rnn):
        """Test MDN loss calculation."""
        batch_size = 4

        z_true = torch.randn(batch_size, 16)
        pi = torch.softmax(torch.randn(batch_size, 3), dim=-1)
        mu = torch.randn(batch_size, 3, 16)
        sigma = torch.abs(torch.randn(batch_size, 3, 16)) + 0.1

        loss, loss_dict = mdn_rnn.loss(z_true, pi, mu, sigma)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert 'mdn_loss' in loss_dict
        assert 'mean_nll' in loss_dict

    def test_init_hidden(self, mdn_rnn):
        """Test hidden state initialization."""
        batch_size = 8
        device = torch.device('cpu')

        # LSTM
        hidden = mdn_rnn.init_hidden(batch_size, device)
        assert isinstance(hidden, tuple)
        assert hidden[0].shape == (1, batch_size, 64)
        assert hidden[1].shape == (1, batch_size, 64)

        # Test GRU
        mdn_rnn.config.rnn_type = "gru"
        mdn_rnn.rnn = torch.nn.GRU(64, 64, 1, batch_first=True)
        hidden = mdn_rnn.init_hidden(batch_size, device)
        assert isinstance(hidden, torch.Tensor)
        assert hidden.shape == (1, batch_size, 64)


class TestWorldModelAgent:
    """Test the integrated WorldModelAgent."""

    @pytest.fixture
    def world_model_agent(self):
        """Create a test WorldModelAgent."""
        return WorldModelAgent(
            observation_shape=(3, 64, 64),
            action_dim=4,
            vae_config=VAEConfig(
                input_channels=3,
                input_height=64,
                input_width=64,
                latent_dim=8
            ),
            mdn_config=MDNRNNConfig(
                latent_dim=8,
                action_dim=4,
                hidden_dim=32,
                num_mixtures=2
            ),
            device='cpu'
        )

    def test_agent_initialization(self, world_model_agent):
        """Test agent initialization."""
        assert world_model_agent.observation_shape == (3, 64, 64)
        assert world_model_agent.action_dim == 4
        assert world_model_agent.vae_config.latent_dim == 8
        assert world_model_agent.mdn_config.hidden_dim == 32

    def test_agent_reset(self, world_model_agent):
        """Test agent reset."""
        world_model_agent.reset(batch_size=2)

        assert world_model_agent.current_hidden is not None
        assert world_model_agent.current_latent is None

    def test_encode_observation(self, world_model_agent):
        """Test observation encoding."""
        obs = torch.randn(1, 3, 64, 64)

        z = world_model_agent.encode_observation(obs)
        assert z.shape == (1, 8)

    def test_agent_act(self, world_model_agent):
        """Test agent action selection."""
        world_model_agent.reset(batch_size=1)

        obs = torch.randn(1, 3, 64, 64)
        action, log_prob, components = world_model_agent.act(
            obs,
            deterministic=False,
            return_components=True
        )

        assert isinstance(action, torch.Tensor)
        assert isinstance(log_prob, torch.Tensor)
        assert components is not None
        assert 'latent' in components
        assert 'hidden' in components
        assert 'controller_input' in components

    def test_imagine_trajectory(self, world_model_agent):
        """Test trajectory imagination."""
        obs = torch.randn(2, 3, 64, 64)
        horizon = 10

        trajectory = world_model_agent.imagine_trajectory(
            obs,
            horizon=horizon,
            temperature=1.0
        )

        assert 'latents' in trajectory
        assert 'actions' in trajectory
        assert trajectory['latents'].shape == (2, horizon + 1, 8)
        assert trajectory['actions'].shape == (2, horizon)

    def test_save_load_components(self, world_model_agent):
        """Test saving and loading model components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            # Save components
            world_model_agent.save_components(save_path)

            # Check files exist
            assert (save_path / 'vae.pt').exists()
            assert (save_path / 'mdn_rnn.pt').exists()
            assert (save_path / 'controller.pt').exists()
            assert (save_path / 'world_model_config.pt').exists()

            # Create new agent and load
            new_agent = WorldModelAgent(
                observation_shape=(3, 64, 64),
                action_dim=4,
                device='cpu'
            )
            new_agent.load_components(save_path)

            # Test that loaded agent works
            obs = torch.randn(1, 3, 64, 64)
            new_agent.reset()
            action, log_prob, _ = new_agent.act(obs)
            assert isinstance(action, torch.Tensor)


class TestIntegration:
    """Integration tests for the world model system."""

    def test_end_to_end_forward_pass(self):
        """Test complete forward pass through all components."""
        # Create agent
        agent = WorldModelAgent(
            observation_shape=(1, 64, 64),
            action_dim=2,
            vae_config=VAEConfig(
                input_channels=1,
                input_height=64,
                input_width=64,
                latent_dim=4,
                hidden_channels=[8, 16]
            ),
            mdn_config=MDNRNNConfig(
                latent_dim=4,
                action_dim=2,
                hidden_dim=16,
                num_mixtures=2
            ),
            device='cpu'
        )

        # Reset agent
        agent.reset(batch_size=1)

        # Run multiple steps
        obs = torch.randn(1, 1, 64, 64)
        for _ in range(5):
            action, log_prob, components = agent.act(
                obs,
                return_components=True
            )

            # Verify outputs
            assert action.shape == torch.Size([]) or action.shape == (1,)
            assert log_prob.shape == torch.Size([]) or log_prob.shape == (1,)
            assert components['latent'].shape == (1, 4)

            # Generate new observation (simplified)
            obs = torch.randn(1, 1, 64, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
