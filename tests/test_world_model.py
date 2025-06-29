# tests/test_world_model.py
"""
Integration tests for World Model PPO implementation.

Tests the VAE, MDN-RNN, and WorldModelPPOTrainer components
to ensure they work correctly together.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
import gym

from janus.agents.components.vae import VariationalAutoencoder
from janus.agents.components.mdn_rnn import MDNRNN
from janus.training.ppo.world_model_trainer import WorldModelPPOTrainer, WorldModelConfig
from janus.training.ppo.config import PPOConfig
from janus.agents.ppo_agent import PPOAgent, NetworkConfig


class TestVAE:
    """Test VariationalAutoencoder component."""

    @pytest.fixture
    def vae(self):
        """Create test VAE instance."""
        return VariationalAutoencoder(
            input_dim=100,
            latent_dim=16,
            hidden_dims=[64, 32],
            beta=1.0
        )

    def test_initialization(self, vae):
        """Test VAE initialization."""
        assert vae.input_dim == 100
        assert vae.latent_dim == 16
        assert vae.beta == 1.0

    def test_forward_pass(self, vae):
        """Test VAE forward pass."""
        batch_size = 8
        x = torch.randn(batch_size, 100)

        recon_x, mu, logvar, z = vae(x)

        assert recon_x.shape == (batch_size, 100)
        assert mu.shape == (batch_size, 16)
        assert logvar.shape == (batch_size, 16)
        assert z.shape == (batch_size, 16)

    def test_loss_calculation(self, vae):
        """Test VAE loss calculation."""
        batch_size = 8
        x = torch.randn(batch_size, 100)
        recon_x, mu, logvar, z = vae(x)

        loss, loss_dict = vae.loss(x, recon_x, mu, logvar)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert 'reconstruction' in loss_dict
        assert 'kl_divergence' in loss_dict
        assert loss_dict['total'] > 0

    def test_encode_decode(self, vae):
        """Test encoding and decoding separately."""
        x = torch.randn(4, 100)

        # Encode
        mu, logvar = vae.encode(x)
        assert mu.shape == (4, 16)
        assert logvar.shape == (4, 16)

        # Sample
        z = vae.reparameterize(mu, logvar)
        assert z.shape == (4, 16)

        # Decode
        recon = vae.decode(z)
        assert recon.shape == (4, 100)

    def test_deterministic_mode(self, vae):
        """Test VAE in eval mode (deterministic)."""
        vae.eval()
        x = torch.randn(1, 100)

        # Multiple forward passes should give same z
        _, mu1, _, z1 = vae(x)
        _, mu2, _, z2 = vae(x)

        assert torch.allclose(z1, mu1)  # In eval, z = mu
        assert torch.allclose(z1, z2)


class TestMDNRNN:
    """Test MDN-RNN component."""

    @pytest.fixture
    def mdn_rnn(self):
        """Create test MDN-RNN instance."""
        return MDNRNN(
            latent_dim=16,
            action_dim=4,
            hidden_dim=64,
            num_mixtures=3
        )

    def test_initialization(self, mdn_rnn):
        """Test MDN-RNN initialization."""
        assert mdn_rnn.latent_dim == 16
        assert mdn_rnn.action_dim == 4
        assert mdn_rnn.hidden_dim == 64
        assert mdn_rnn.num_mixtures == 3

    def test_forward_single_step(self, mdn_rnn):
        """Test single step prediction."""
        batch_size = 4
        latent = torch.randn(batch_size, 16)
        action = torch.randn(batch_size, 4)
        hidden = mdn_rnn.init_hidden(batch_size)

        pi, mu, log_sigma, new_hidden = mdn_rnn(latent, action, hidden)

        assert pi.shape == (batch_size, 3)  # num_mixtures
        assert mu.shape == (batch_size, 3, 16)  # num_mixtures, latent_dim
        assert log_sigma.shape == (batch_size, 3, 16)
        assert new_hidden[0].shape == (1, batch_size, 64)  # LSTM hidden

    def test_forward_sequence(self, mdn_rnn):
        """Test sequence prediction."""
        batch_size = 2
        seq_len = 5
        latent_seq = torch.randn(batch_size, seq_len, 16)
        action_seq = torch.randn(batch_size, seq_len, 4)

        pi, mu, log_sigma, hidden = mdn_rnn(latent_seq, action_seq)

        assert pi.shape == (batch_size, seq_len, 3)
        assert mu.shape == (batch_size, seq_len, 3, 16)
        assert log_sigma.shape == (batch_size, seq_len, 3, 16)

    def test_sampling(self, mdn_rnn):
        """Test sampling from mixture distribution."""
        batch_size = 8
        pi = torch.softmax(torch.randn(batch_size, 3), dim=-1)
        mu = torch.randn(batch_size, 3, 16)
        log_sigma = torch.randn(batch_size, 3, 16) * 0.1

        z_next = mdn_rnn.sample(pi, mu, log_sigma)
        assert z_next.shape == (batch_size, 16)

    def test_loss_calculation(self, mdn_rnn):
        """Test MDN loss calculation."""
        batch_size = 4
        z_true = torch.randn(batch_size, 16)
        pi = torch.softmax(torch.randn(batch_size, 3), dim=-1)
        mu = torch.randn(batch_size, 3, 16)
        log_sigma = torch.randn(batch_size, 3, 16) * 0.1

        loss, loss_dict = mdn_rnn.loss(z_true, pi, mu, log_sigma)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert 'nll' in loss_dict
        assert loss_dict['nll'] > 0


class TestWorldModelPPOTrainer:
    """Test WorldModelPPOTrainer integration."""

    @pytest.fixture
    def setup(self):
        """Setup test environment and configurations."""
        # Create simple environment
        envs = [gym.make('CartPole-v1') for _ in range(2)]

        # Configurations
        ppo_config = PPOConfig(
            learning_rate=3e-4,
            n_epochs=2,
            batch_size=32,
            num_workers=2,
            log_interval=100
        )

        wm_config = WorldModelConfig(
            vae_latent_dim=8,
            vae_hidden_dims=[32],
            mdn_hidden_dim=32,
            mdn_num_mixtures=2,
            pretrain_epochs=1,
            pretrain_batch_size=16,
            random_collection_steps=100,
            imagination_ratio=0.3,
            imagination_horizon=5
        )

        # Create agent
        agent = PPOAgent(
            observation_dim=4,
            action_dim=2,
            actor_config=NetworkConfig(layer_sizes=[32]),
            critic_config=NetworkConfig(layer_sizes=[32])
        )

        return envs, ppo_config, wm_config, agent

    def test_trainer_initialization(self, setup):
        """Test trainer initialization."""
        envs, ppo_config, wm_config, agent = setup

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = WorldModelPPOTrainer(
                agent=agent,
                envs=envs,
                config=ppo_config,
                world_model_config=wm_config,
                checkpoint_dir=Path(tmpdir),
                use_tensorboard=False,
                use_wandb=False
            )

            assert trainer.vae is not None
            assert trainer.mdn_rnn is not None
            assert trainer.vae.latent_dim == 8
            assert trainer.mdn_rnn.latent_dim == 8

    def test_random_experience_collection(self, setup):
        """Test collecting random experience."""
        envs, ppo_config, wm_config, agent = setup

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = WorldModelPPOTrainer(
                agent=agent,
                envs=envs,
                config=ppo_config,
                world_model_config=wm_config,
                checkpoint_dir=Path(tmpdir),
                use_tensorboard=False,
                use_wandb=False
            )

            # Collect experience
            trainer.collect_random_experience(50)

            assert len(trainer.experience_buffer['observations']) >= 50
            assert len(trainer.experience_buffer['actions']) >= 50
            assert len(trainer.experience_buffer['next_observations']) >= 50

    def test_world_model_pretraining(self, setup):
        """Test pretraining VAE and MDN-RNN."""
        envs, ppo_config, wm_config, agent = setup

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = WorldModelPPOTrainer(
                agent=agent,
                envs=envs,
                config=ppo_config,
                world_model_config=wm_config,
                checkpoint_dir=Path(tmpdir),
                use_tensorboard=False,
                use_wandb=False
            )

            # Collect and pretrain
            trainer.collect_random_experience(100)
            trainer.pretrain_world_model()

            # Test that models can encode observations
            obs = torch.randn(1, 4)
            with torch.no_grad():
                z, _ = trainer.vae.encode(obs.view(1, -1))
                assert z.shape == (1, 8)

    def test_imagination_rollout(self, setup):
        """Test generating imagined trajectories."""
        envs, ppo_config, wm_config, agent = setup

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = WorldModelPPOTrainer(
                agent=agent,
                envs=envs,
                config=ppo_config,
                world_model_config=wm_config,
                checkpoint_dir=Path(tmpdir),
                use_tensorboard=False,
                use_wandb=False
            )

            # Need some experience first
            trainer.collect_random_experience(100)
            trainer.pretrain_world_model()

            # Test imagination
            start_obs = torch.randn(2, 4)
            imagined = trainer.imagine_rollout(start_obs, horizon=5)

            assert 'latents' in imagined
            assert 'actions' in imagined
            assert 'values' in imagined
            assert imagined['latents'].shape == (2, 5, 8)
            assert imagined['actions'].shape == (2, 5)
            assert imagined['values'].shape == (2, 5)

    def test_mixed_rollout_collection(self, setup):
        """Test collecting both real and imagined rollouts."""
        envs, ppo_config, wm_config, agent = setup

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = WorldModelPPOTrainer(
                agent=agent,
                envs=envs,
                config=ppo_config,
                world_model_config=wm_config,
                checkpoint_dir=Path(tmpdir),
                use_tensorboard=False,
                use_wandb=False
            )

            # Initialize buffer
            from janus.training.ppo.buffer import RolloutBuffer
            trainer.buffer = RolloutBuffer(
                buffer_size=100,
                obs_shape=(wm_config.vae_latent_dim + wm_config.mdn_hidden_dim,),
                action_shape=(),
                device=trainer.device,
                n_envs=trainer.n_envs
            )

            # Collect experience and pretrain
            trainer.collect_random_experience(200)
            trainer.pretrain_world_model()

            # Collect mixed rollouts
            metrics = trainer.collect_rollouts(50)

            assert 'mean_episode_reward' in metrics
            assert trainer.buffer.ptr > 0

    def test_checkpoint_save_load(self, setup):
        """Test saving and loading world model checkpoints."""
        envs, ppo_config, wm_config, agent = setup

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create and train first trainer
            trainer1 = WorldModelPPOTrainer(
                agent=agent,
                envs=envs,
                config=ppo_config,
                world_model_config=wm_config,
                checkpoint_dir=tmpdir,
                use_tensorboard=False,
                use_wandb=False
            )

            trainer1.collect_random_experience(100)
            trainer1.pretrain_world_model()

            # Save checkpoint
            checkpoint_path = trainer1.save_checkpoint()

            # Create new trainer and load
            trainer2 = WorldModelPPOTrainer(
                agent=PPOAgent(
                    observation_dim=4,
                    action_dim=2,
                    actor_config=NetworkConfig(layer_sizes=[32]),
                    critic_config=NetworkConfig(layer_sizes=[32])
                ),
                envs=envs,
                config=ppo_config,
                world_model_config=wm_config,
                checkpoint_dir=tmpdir,
                use_tensorboard=False,
                use_wandb=False
            )

            trainer2.load_checkpoint(checkpoint_path)

            # Test that loaded models produce same outputs
            test_obs = torch.randn(1, 4)
            with torch.no_grad():
                z1, _ = trainer1.vae.encode(test_obs.view(1, -1))
                z2, _ = trainer2.vae.encode(test_obs.view(1, -1))
                assert torch.allclose(z1, z2, atol=1e-6)


class TestIntegration:
    """Full integration tests."""

    def test_short_training_run(self):
        """Test a complete but short training run."""
        # Setup
        envs = [gym.make('CartPole-v1') for _ in range(2)]

        ppo_config = PPOConfig(
            learning_rate=3e-4,
            n_epochs=1,
            batch_size=16,
            log_interval=10
        )

        wm_config = WorldModelConfig(
            vae_latent_dim=4,
            vae_hidden_dims=[16],
            mdn_hidden_dim=16,
            pretrain_epochs=1,
            random_collection_steps=50,
            imagination_ratio=0.2
        )

        agent = PPOAgent(
            observation_dim=4,
            action_dim=2,
            actor_config=NetworkConfig(layer_sizes=[16]),
            critic_config=NetworkConfig(layer_sizes=[16])
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = WorldModelPPOTrainer(
                agent=agent,
                envs=envs,
                config=ppo_config,
                world_model_config=wm_config,
                checkpoint_dir=Path(tmpdir),
                use_tensorboard=False,
                use_wandb=False
            )

            # Run short training
            trainer.train(total_timesteps=500, rollout_length=100)

            # Check that training progressed
            assert trainer.global_step >= 500
            assert trainer.num_updates > 0

            # Check that world model was used
            assert len(trainer.experience_buffer['observations']) > 0


if __name__ == '__main__':
    # Run a quick integration test
    print("Running World Model integration tests...")

    # Test VAE
    print("\n1. Testing VAE...")
    vae = VariationalAutoencoder(input_dim=10, latent_dim=4)
    x = torch.randn(2, 10)
    recon, mu, logvar, z = vae(x)
    print(f"✓ VAE forward pass: {x.shape} -> {z.shape}")

    # Test MDN-RNN
    print("\n2. Testing MDN-RNN...")
    mdn = MDNRNN(latent_dim=4, action_dim=2)
    action = torch.randn(2, 2)
    pi, mu, log_sigma, hidden = mdn(z, action)
    print(f"✓ MDN-RNN forward pass: z={z.shape}, a={action.shape} -> pi={pi.shape}")

    # Test trainer
    print("\n3. Testing WorldModelPPOTrainer...")
    test = TestIntegration()
    test.test_short_training_run()
    print("✓ Short training run completed")

    print("\nAll integration tests passed! ✓")
