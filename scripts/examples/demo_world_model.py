# scripts/demo_world_model.py
"""
Interactive demo of the World Model showing:
1. Observation encoding/decoding with VAE
2. Future state prediction with MDN-RNN
3. Trajectory imagination
4. Comparison of real vs imagined rollouts
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import gymnasium as gym
from typing import List, Tuple

from janus.agents.world_model_agent import WorldModelAgent
from janus.agents.components.vae import VAEConfig, preprocess_observation
from janus.agents.components.mdn_rnn import MDNRNNConfig
from janus.training.world_model.train_world_model import WorldModelTrainer, WorldModelTrainingConfig


class WorldModelDemo:
    """Interactive demonstration of World Model capabilities."""

    def __init__(self, env_name: str = "CarRacing-v2", device: str = "cuda"):
        """
        Initialize the demo.

        Args:
            env_name: Gymnasium environment name
            device: Device to run on
        """
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Get environment info
        obs, _ = self.env.reset()
        self.obs_shape = obs.shape
        self.action_dim = self.env.action_space.shape[0] if hasattr(self.env.action_space, 'shape') else self.env.action_space.n
        self.continuous = hasattr(self.env.action_space, 'shape')

        print(f"Environment: {env_name}")
        print(f"Observation shape: {self.obs_shape}")
        print(f"Action dimension: {self.action_dim}")
        print(f"Continuous actions: {self.continuous}")

    def train_or_load_world_model(self, checkpoint_dir: Path = Path("checkpoints/world_model_demo")) -> WorldModelAgent:
        """
        Train a new world model or load existing one.

        Args:
            checkpoint_dir: Directory for checkpoints

        Returns:
            Trained WorldModelAgent
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Check if pre-trained model exists
        if (checkpoint_dir / "vae_final.pt").exists() and (checkpoint_dir / "mdn_final.pt").exists():
            print("Loading pre-trained world model...")

            # Load config
            config_data = torch.load(checkpoint_dir / "world_model_complete.pt")

            # Create agent
            agent = WorldModelAgent(
                observation_shape=config_data['observation_shape'],
                action_dim=config_data['action_dim'],
                vae_config=config_data['vae_config'],
                mdn_config=config_data['mdn_config'],
                continuous_actions=config_data['continuous_actions'],
                device=self.device
            )

            # Load weights
            agent.vae.load_state_dict(torch.load(checkpoint_dir / "vae_final.pt"))
            agent.mdn_rnn.load_state_dict(torch.load(checkpoint_dir / "mdn_final.pt"))

        else:
            print("Training new world model...")

            # Configure training
            config = WorldModelTrainingConfig(
                num_episodes=100,  # Reduced for demo
                vae_epochs=20,
                mdn_epochs=20,
                device=str(self.device),
                checkpoint_dir=str(checkpoint_dir)
            )

            # Train
            trainer = WorldModelTrainer(self.env, config)
            vae, mdn_rnn = trainer.train_world_model()

            # Create agent
            agent = WorldModelAgent(
                observation_shape=trainer.observation_shape,
                action_dim=trainer.action_dim,
                continuous_actions=trainer.continuous_actions,
                device=self.device
            )

            # Load trained components
            agent.vae.load_state_dict(vae.state_dict())
            agent.mdn_rnn.load_state_dict(mdn_rnn.state_dict())

        return agent

    def demo_vae_reconstruction(self, agent: WorldModelAgent, num_samples: int = 5):
        """
        Demonstrate VAE reconstruction capability.

        Args:
            agent: Trained WorldModelAgent
            num_samples: Number of samples to show
        """
        print("\n=== VAE Reconstruction Demo ===")

        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        fig.suptitle("VAE Reconstruction: Original (top) vs Reconstructed (bottom)")

        for i in range(num_samples):
            # Get observation
            obs, _ = self.env.reset()
            obs_tensor = preprocess_observation(obs).to(self.device)

            # Encode and decode
            with torch.no_grad():
                reconstruction, mu, logvar = agent.vae(obs_tensor)

            # Convert to numpy for display
            original = obs_tensor.squeeze().cpu().numpy()
            reconstructed = reconstruction.squeeze().cpu().numpy()

            # Handle different image formats
            if original.shape[0] in [1, 3]:  # Channel first
                original = np.transpose(original, (1, 2, 0))
                reconstructed = np.transpose(reconstructed, (1, 2, 0))

            # Display
            axes[0, i].imshow(original.squeeze())
            axes[0, i].axis('off')
            axes[0, i].set_title(f"Original {i+1}")

            axes[1, i].imshow(reconstructed.squeeze())
            axes[1, i].axis('off')
            axes[1, i].set_title(f"Reconstructed {i+1}")

        plt.tight_layout()
        plt.show()

        print(f"Latent dimension: {agent.vae_config.latent_dim}")
        print(f"Compression ratio: {np.prod(self.obs_shape) / agent.vae_config.latent_dim:.1f}x")

    def demo_future_prediction(self, agent: WorldModelAgent, num_steps: int = 20):
        """
        Demonstrate future state prediction with MDN-RNN.

        Args:
            agent: Trained WorldModelAgent
            num_steps: Number of steps to predict
        """
        print("\n=== Future State Prediction Demo ===")

        # Reset environment and agent
        obs, _ = self.env.reset()
        agent.reset()

        # Storage
        real_observations = [obs]
        predicted_latents = []
        actions = []

        # Initial encoding
        obs_tensor = preprocess_observation(obs).to(self.device)
        z = agent.encode_observation(obs_tensor, deterministic=True)
        predicted_latents.append(z)

        print(f"Collecting {num_steps} steps of real data and predictions...")

        for step in range(num_steps):
            # Get action from agent
            action, _, _ = agent.act(obs_tensor)

            # Convert action for environment
            if self.continuous:
                action_np = action.cpu().numpy()
            else:
                action_np = action.cpu().item()

            # Step real environment
            obs, reward, done, truncated, _ = self.env.step(action_np)
            real_observations.append(obs)

            # Predict next latent state
            with torch.no_grad():
                if self.continuous:
                    action_tensor = action
                else:
                    action_one_hot = torch.zeros(1, self.action_dim, device=self.device)
                    action_one_hot[0, action] = 1
                    action_tensor = action_one_hot

                # Get prediction from MDN-RNN
                pi, mu, sigma, _ = agent.mdn_rnn(
                    z.unsqueeze(1),
                    action_tensor.unsqueeze(1),
                    agent.current_hidden
                )

                # Sample next latent
                z_next = agent.mdn_rnn.sample(pi.squeeze(1), mu.squeeze(1), sigma.squeeze(1))
                predicted_latents.append(z_next)
                z = z_next

            actions.append(action_np)

            if done or truncated:
                break

        # Decode predicted latents back to observations
        predicted_observations = []
        with torch.no_grad():
            for z in predicted_latents:
                decoded = agent.vae.decode(z)
                predicted_observations.append(decoded.squeeze().cpu().numpy())

        # Visualize comparison
        self._visualize_prediction_comparison(real_observations, predicted_observations)

    def _visualize_prediction_comparison(self, real_obs: List[np.ndarray],
                                       predicted_obs: List[np.ndarray]):
        """Visualize real vs predicted observations."""
        num_steps = min(len(real_obs), len(predicted_obs), 10)

        fig, axes = plt.subplots(2, num_steps, figsize=(20, 6))
        fig.suptitle("Real (top) vs Predicted (bottom) Observations")

        for i in range(num_steps):
            # Real observation
            real = real_obs[i]
            axes[0, i].imshow(real)
            axes[0, i].axis('off')
            axes[0, i].set_title(f"Real t={i}")

            # Predicted observation
            pred = predicted_obs[i]
            if pred.shape[0] in [1, 3]:  # Channel first
                pred = np.transpose(pred, (1, 2, 0))
            axes[1, i].imshow(pred.squeeze())
            axes[1, i].axis('off')
            axes[1, i].set_title(f"Predicted t={i}")

        plt.tight_layout()
        plt.show()

    def demo_trajectory_imagination(self, agent: WorldModelAgent, num_trajectories: int = 3):
        """
        Demonstrate trajectory imagination capability.

        Args:
            agent: Trained WorldModelAgent
            num_trajectories: Number of imagined trajectories to show
        """
        print("\n=== Trajectory Imagination Demo ===")

        # Get starting observation
        obs, _ = self.env.reset()
        obs_tensor = preprocess_observation(obs).to(self.device)

        fig, axes = plt.subplots(num_trajectories, 10, figsize=(20, 6))
        fig.suptitle("Imagined Trajectories from Same Starting State")

        for traj_idx in range(num_trajectories):
            # Imagine trajectory with different temperature (exploration)
            temperature = 0.5 + traj_idx * 0.5
            trajectory = agent.imagine_trajectory(
                obs_tensor,
                horizon=9,
                temperature=temperature
            )

            # Decode latents to observations
            latents = trajectory['latents'][0]  # Remove batch dimension

            with torch.no_grad():
                for step in range(min(10, latents.shape[0])):
                    decoded = agent.vae.decode(latents[step:step+1])
                    decoded_np = decoded.squeeze().cpu().numpy()

                    if decoded_np.shape[0] in [1, 3]:  # Channel first
                        decoded_np = np.transpose(decoded_np, (1, 2, 0))

                    axes[traj_idx, step].imshow(decoded_np.squeeze())
                    axes[traj_idx, step].axis('off')
                    if step == 0:
                        axes[traj_idx, step].set_ylabel(f"Temp={temperature:.1f}", rotation=0, labelpad=40)

        plt.tight_layout()
        plt.show()

        print(f"Generated {num_trajectories} imagined trajectories with different exploration levels")

    def create_animation(self, agent: WorldModelAgent, num_steps: int = 100, save_path: str = "world_model_demo.gif"):
        """
        Create an animation showing real vs imagined rollout.

        Args:
            agent: Trained WorldModelAgent
            num_steps: Number of steps to animate
            save_path: Path to save animation
        """
        print(f"\n=== Creating Animation ({num_steps} steps) ===")

        # Collect real trajectory
        obs, _ = self.env.reset()
        agent.reset()

        real_frames = []
        imagined_frames = []

        # Initial state
        real_frames.append(self.env.render())

        # Imagine entire trajectory upfront
        obs_tensor = preprocess_observation(obs).to(self.device)
        trajectory = agent.imagine_trajectory(obs_tensor, horizon=num_steps)

        # Decode imagined trajectory
        with torch.no_grad():
            latents = trajectory['latents'][0]  # Remove batch dimension
            for i in range(min(num_steps, latents.shape[0])):
                decoded = agent.vae.decode(latents[i:i+1])
                decoded_np = decoded.squeeze().cpu().numpy()

                if decoded_np.shape[0] in [1, 3]:  # Channel first
                    decoded_np = np.transpose(decoded_np, (1, 2, 0))

                # Convert to uint8 for display
                decoded_np = (decoded_np * 255).astype(np.uint8)
                imagined_frames.append(decoded_np.squeeze())

        # Collect real trajectory
        for step in range(num_steps):
            # Random action for demo
            action = self.env.action_space.sample()
            obs, _, done, truncated, _ = self.env.step(action)
            real_frames.append(self.env.render())

            if done or truncated:
                obs, _ = self.env.reset()

        # Create side-by-side animation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle("Real Environment vs Imagined Trajectory")

        ax1.set_title("Real")
        ax1.axis('off')
        ax2.set_title("Imagined")
        ax2.axis('off')

        im1 = ax1.imshow(real_frames[0])
        im2 = ax2.imshow(imagined_frames[0])

        def animate(frame):
            if frame < len(real_frames):
                im1.set_array(real_frames[frame])
            if frame < len(imagined_frames):
                im2.set_array(imagined_frames[frame])
            return [im1, im2]

        anim = animation.FuncAnimation(
            fig, animate, frames=max(len(real_frames), len(imagined_frames)),
            interval=50, blit=True
        )

        anim.save(save_path, writer='pillow')
        print(f"Animation saved to {save_path}")

        plt.close()


def main():
    """Run the complete world model demo."""
    import argparse

    parser = argparse.ArgumentParser(description="World Model Demo")
    parser.add_argument("--env", type=str, default="CarRacing-v2", help="Environment name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--train", action="store_true", help="Force training new model")
    parser.add_argument("--no-animation", action="store_true", help="Skip animation creation")
    args = parser.parse_args()

    # Create demo
    demo = WorldModelDemo(env_name=args.env, device=args.device)

    # Train or load model
    checkpoint_dir = Path(f"checkpoints/world_model_demo_{args.env.replace('-', '_')}")
    if args.train and checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)

    agent = demo.train_or_load_world_model(checkpoint_dir)

    # Run demos
    demo.demo_vae_reconstruction(agent)
    demo.demo_future_prediction(agent)
    demo.demo_trajectory_imagination(agent)

    if not args.no_animation:
        demo.create_animation(agent)

    print("\n=== Demo Complete ===")
    print("The World Model can now be used for:")
    print("1. Compressed representation learning")
    print("2. Future state prediction")
    print("3. Planning through imagination")
    print("4. Data-efficient reinforcement learning")


if __name__ == "__main__":
    main()
