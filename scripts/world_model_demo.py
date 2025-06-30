# scripts/examples/world_model_demo.py
"""
Interactive demonstration of the World Model PPO implementation.

This script shows how the World Model learns to compress observations
and predict future states, enabling the agent to "dream" and learn
from imagined experience.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gym
from pathlib import Path

from janus.agents.components.vae import VariationalAutoencoder
from janus.agents.components.mdn_rnn import MDNRNN
from janus.training.ppo.world_model_trainer import (
    WorldModelPPOTrainer,
    WorldModelConfig,
)
from janus.training.ppo.config import PPOConfig
from janus.agents.ppo_agent import PPOAgent, NetworkConfig


def visualize_vae_reconstruction(vae, observations, title="VAE Reconstruction"):
    """Visualize how well the VAE reconstructs observations."""
    vae.eval()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(title)

    # Select 5 random observations
    indices = np.random.choice(len(observations), 5, replace=False)

    for i, idx in enumerate(indices):
        obs = observations[idx]
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        # Reconstruct
        with torch.no_grad():
            obs_flat = obs_tensor.view(1, -1)
            recon, mu, logvar, z = vae(obs_flat)
            recon = recon.view(obs.shape)

        # Original
        axes[0, i].bar(range(len(obs)), obs)
        axes[0, i].set_title(f"Original {idx}")
        axes[0, i].set_ylim(-3, 3)

        # Reconstruction
        axes[1, i].bar(range(len(obs)), recon.numpy())
        axes[1, i].set_title(f"Reconstructed (z_dim={z.shape[1]})")
        axes[1, i].set_ylim(-3, 3)

    plt.tight_layout()
    return fig


def visualize_latent_space(vae, observations, labels=None):
    """Visualize the learned latent space (2D projection)."""
    vae.eval()

    # Encode all observations
    latents = []
    with torch.no_grad():
        for obs in observations:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).view(1, -1)
            mu, _ = vae.encode(obs_tensor)
            latents.append(mu.numpy())

    latents = np.concatenate(latents, axis=0)

    # Plot first 2 dimensions
    fig, ax = plt.subplots(figsize=(8, 8))

    if labels is not None:
        scatter = ax.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Episode Progress')
    else:
        ax.scatter(latents[:, 0], latents[:, 1], alpha=0.6)

    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title('Learned Latent Space (2D Projection)')
    ax.grid(True, alpha=0.3)

    return fig


def visualize_imagination_vs_reality(trainer, start_obs, actual_trajectory, horizon=10):
    """Compare imagined trajectory with actual environment rollout."""
    trainer.vae.eval()
    trainer.mdn_rnn.eval()

    # Generate imagined trajectory
    start_tensor = torch.FloatTensor(start_obs).unsqueeze(0)
    imagined = trainer.imagine_rollout(start_tensor, horizon=horizon)

    # Decode imagined latents back to observation space
    imagined_obs = []
    with torch.no_grad():
        for t in range(horizon):
            z = imagined['latents'][0, t].unsqueeze(0)
            recon = trainer.vae.decode(z)
            imagined_obs.append(recon.numpy().reshape(start_obs.shape))

    # Plot comparison
    fig, axes = plt.subplots(2, horizon, figsize=(3*horizon, 6))
    fig.suptitle('Imagination vs Reality')

    for t in range(horizon):
        # Reality
        if t < len(actual_trajectory):
            axes[0, t].bar(range(len(actual_trajectory[t])), actual_trajectory[t])
            axes[0, t].set_title(f'Real t={t}')
            axes[0, t].set_ylim(-3, 3)

        # Imagination
        axes[1, t].bar(range(len(imagined_obs[t])), imagined_obs[t])
        axes[1, t].set_title(f'Imagined t={t}')
        axes[1, t].set_ylim(-3, 3)

    plt.tight_layout()
    return fig


def run_demo():
    """Run the complete World Model demo."""
    print("=== World Model Demo ===\n")

    # 1. Setup environment
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Environment: CartPole-v1")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}\n")

    # 2. Collect some data
    print("Collecting demonstration data...")
    observations = []
    actions = []
    episode_starts = []

    for episode in range(20):
        obs = env.reset()
        episode_starts.append(len(observations))

        for t in range(100):
            observations.append(obs)
            action = env.action_space.sample()
            actions.append(action)

            obs, reward, done, _ = env.step(action)

            if done:
                break

    observations = np.array(observations)
    actions = np.array(actions)
    labels = np.array([i // 10 for i in range(len(observations))])  # Time step labels

    print(f"Collected {len(observations)} observations\n")

    # 3. Create and train VAE
    print("Training VAE...")
    vae = VariationalAutoencoder(
        input_dim=obs_dim,
        latent_dim=8,
        hidden_dims=[32, 16],
        beta=0.5
    )

    # Simple training loop
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae.train()

    for epoch in range(50):
        # Random batch
        batch_idx = np.random.choice(len(observations), 32)
        batch_obs = torch.FloatTensor(observations[batch_idx])

        # Forward pass
        recon, mu, logvar, z = vae(batch_obs)
        loss, _ = vae.loss(batch_obs, recon, mu, logvar)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")

    print("\n4. Visualizing VAE performance...")

    # Visualize reconstruction
    fig1 = visualize_vae_reconstruction(vae, observations[:20], "VAE Reconstruction Quality")
    plt.savefig('vae_reconstruction.png')
    print("  Saved: vae_reconstruction.png")

    # Visualize latent space
    fig2 = visualize_latent_space(vae, observations[:200], labels[:200])
    plt.savefig('latent_space.png')
    print("  Saved: latent_space.png")

    # 5. Create and train MDN-RNN
    print("\n5. Training MDN-RNN...")
    mdn_rnn = MDNRNN(
        latent_dim=8,
        action_dim=action_dim,
        hidden_dim=32,
        num_mixtures=3
    )

    # Encode observations
    latents = []
    with torch.no_grad():
        for obs in observations:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            mu, _ = vae.encode(obs_tensor)
            latents.append(mu)
    latents = torch.cat(latents, dim=0)

    # Train MDN-RNN
    mdn_optimizer = torch.optim.Adam(mdn_rnn.parameters(), lr=1e-3)
    mdn_rnn.train()

    for epoch in range(50):
        # Create sequence batch
        seq_len = 10
        batch_size = 8

        total_loss = 0
        num_batches = 0

        for _ in range(5):  # 5 batches per epoch
            # Random starting points
            start_idx = np.random.choice(len(latents) - seq_len - 1, batch_size)

            # Create sequences
            z_seq = torch.stack([latents[i:i+seq_len] for i in start_idx])
            a_seq = torch.FloatTensor([[actions[i+t] for t in range(seq_len)] for i in start_idx])
            z_next_seq = torch.stack([latents[i+1:i+seq_len+1] for i in start_idx])

            # Forward pass
            hidden = mdn_rnn.init_hidden(batch_size)
            pi, mu, log_sigma, _ = mdn_rnn(z_seq, a_seq, hidden)

            # Calculate loss for each timestep
            for t in range(seq_len):
                loss, _ = mdn_rnn.loss(z_next_seq[:, t], pi[:, t], mu[:, t], log_sigma[:, t])
                total_loss += loss
                num_batches += 1

        # Update
        avg_loss = total_loss / num_batches
        mdn_optimizer.zero_grad()
        avg_loss.backward()
        mdn_optimizer.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}, Loss: {avg_loss.item():.4f}")

    # 6. Test imagination
    print("\n6. Testing imagination vs reality...")

    # Collect a real trajectory
    obs = env.reset()
    real_trajectory = [obs]

    for t in range(10):
        action = env.action_space.sample()
        obs, _, done, _ = env.step(action)
        real_trajectory.append(obs)
        if done:
            break

    # Create a simple trainer for imagination
    # (In practice, you'd use the full WorldModelPPOTrainer)
    class SimpleTrainer:
        def __init__(self, vae, mdn_rnn):
            self.vae = vae
            self.mdn_rnn = mdn_rnn

        def imagine_rollout(self, start_obs, horizon):
            self.vae.eval()
            self.mdn_rnn.eval()

            # Encode starting observation
            with torch.no_grad():
                z_current, _ = self.vae.encode(start_obs.view(1, -1))
                hidden = self.mdn_rnn.init_hidden(1)

            imagined_latents = []

            for t in range(horizon):
                with torch.no_grad():
                    # Random action for demo
                    action = torch.tensor([[env.action_space.sample()]], dtype=torch.float32)

                    # Predict next latent
                    pi, mu, log_sigma, hidden = self.mdn_rnn(z_current, action, hidden)
                    z_next = self.mdn_rnn.sample(pi, mu, log_sigma)

                    imagined_latents.append(z_current)
                    z_current = z_next

            return {'latents': torch.stack(imagined_latents).unsqueeze(0)}

    simple_trainer = SimpleTrainer(vae, mdn_rnn)

    # Visualize imagination vs reality
    fig3 = visualize_imagination_vs_reality(
        simple_trainer,
        real_trajectory[0],
        real_trajectory[1:],
        horizon=min(5, len(real_trajectory)-1)
    )
    plt.savefig('imagination_vs_reality.png')
    print("  Saved: imagination_vs_reality.png")

    print("\n=== Demo Complete ===")
    print("\nThe World Model has learned to:")
    print("1. Compress observations into a compact latent space (VAE)")
    print("2. Predict future latent states given actions (MDN-RNN)")
    print("3. Generate imagined trajectories for planning")
    print("\nCheck the generated images to see the results!")

    # Show all plots
    plt.show()


if __name__ == '__main__':
    run_demo()
