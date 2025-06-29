# examples/multi_agent_training_example.py

"""
Complete example demonstrating Multi-Agent PPO with centralized training.
Shows how to integrate all components for different multi-agent scenarios.
"""

import logging
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from janus.agents.multi_agent_ppo import MultiAgentPPOAgent, MultiAgentNetworkConfig
from janus.training.ppo.multi_agent_trainer import MultiAgentPPOTrainer
from janus.training.ppo.multi_agent_config import MultiAgentConfig, create_example_configs
from janus.agents.ppo_agent import NetworkConfig


# Example Multi-Agent Environment Wrapper
class MultiAgentEnvWrapper:
    """
    Wrapper to make single-agent environments work with multi-agent training.
    For demonstration purposes - in practice, use proper multi-agent environments.
    """

    def __init__(self, env_name: str, n_agents: int):
        self.n_agents = n_agents
        self.envs = [gym.make(env_name) for _ in range(n_agents)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self) -> Tuple[List[np.ndarray], Dict]:
        """Reset all environments."""
        observations = []
        infos = []

        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)

        return observations, {"individual_infos": infos}

    def step(self, actions: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[bool], bool, Dict]:
        """
        Step all environments with given actions.

        Args:
            actions: Array of actions [n_agents]

        Returns:
            observations, rewards, dones, truncated, info
        """
        observations = []
        rewards = []
        dones = []
        truncated = False
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, trunc, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            truncated = truncated or trunc
            infos.append(info)

        return observations, rewards, dones, truncated, {"individual_infos": infos}

    def render(self):
        """Render first environment."""
        return self.envs[0].render()

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


# Cooperative Multi-Agent Environment Example
class CooperativeNavigationEnv:
    """
    Simple cooperative navigation task.
    Agents must reach different target locations while avoiding collisions.
    """

    def __init__(self, n_agents: int = 3, world_size: float = 5.0, max_steps: int = 100):
        self.n_agents = n_agents
        self.world_size = world_size
        self.max_steps = max_steps
        self.current_step = 0

        # Observation: [agent_x, agent_y, target_x, target_y, other_agents_relative_positions]
        self.observation_space = gym.spaces.Box(
            low=-self.world_size,
            high=self.world_size,
            shape=(4 + 2 * (n_agents - 1),),
            dtype=np.float32
        )

        # Action: [move_x, move_y] continuous
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        self.reset()

    def reset(self) -> Tuple[List[np.ndarray], Dict]:
        """Reset environment."""
        self.current_step = 0

        # Random initial positions
        self.agent_positions = np.random.uniform(
            -self.world_size/2, self.world_size/2, (self.n_agents, 2)
        )

        # Random target positions
        self.target_positions = np.random.uniform(
            -self.world_size/2, self.world_size/2, (self.n_agents, 2)
        )

        return self._get_observations(), {}

    def _get_observations(self) -> List[np.ndarray]:
        """Get observations for all agents."""
        observations = []

        for i in range(self.n_agents):
            # Own position and target
            obs = np.concatenate([
                self.agent_positions[i],
                self.target_positions[i]
            ])

            # Relative positions of other agents
            for j in range(self.n_agents):
                if i != j:
                    relative_pos = self.agent_positions[j] - self.agent_positions[i]
                    obs = np.concatenate([obs, relative_pos])

            observations.append(obs.astype(np.float32))

        return observations

    def step(self, actions: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[bool], bool, Dict]:
        """
        Step environment.

        Args:
            actions: Actions for all agents [n_agents, 2]
        """
        self.current_step += 1

        # Update positions
        self.agent_positions += actions * 0.1  # Scale down movements

        # Clip to world bounds
        self.agent_positions = np.clip(
            self.agent_positions,
            -self.world_size/2,
            self.world_size/2
        )

        # Calculate rewards
        rewards = []
        dones = []

        for i in range(self.n_agents):
            # Distance to target
            dist_to_target = np.linalg.norm(
                self.agent_positions[i] - self.target_positions[i]
            )

            # Reward for reaching target
            if dist_to_target < 0.5:
                reward = 10.0
                done = True
            else:
                # Negative reward proportional to distance
                reward = -dist_to_target * 0.1
                done = False

            # Collision penalty
            for j in range(self.n_agents):
                if i != j:
                    dist_to_other = np.linalg.norm(
                        self.agent_positions[i] - self.agent_positions[j]
                    )
                    if dist_to_other < 0.5:
                        reward -= 5.0

            rewards.append(reward)
            dones.append(done)

        # Episode ends if all agents reach targets or max steps
        truncated = self.current_step >= self.max_steps

        info = {
            "positions": self.agent_positions.copy(),
            "targets": self.target_positions.copy(),
            "episode_step": self.current_step
        }

        return self._get_observations(), rewards, dones, truncated, info

    def render(self):
        """Simple console rendering."""
        print(f"\nStep {self.current_step}:")
        for i in range(self.n_agents):
            pos = self.agent_positions[i]
            target = self.target_positions[i]
            dist = np.linalg.norm(pos - target)
            print(f"  Agent {i}: pos=({pos[0]:.2f}, {pos[1]:.2f}), "
                  f"target=({target[0]:.2f}, {target[1]:.2f}), dist={dist:.2f}")


def train_cooperative_agents():
    """Train agents in a cooperative navigation task."""
    print("\n=== Training Cooperative Multi-Agent System ===\n")

    # Configuration
    n_agents = 3
    n_envs = 4

    # Create config
    config = MultiAgentConfig(
        # Multi-agent settings
        n_agents=n_agents,
        centralized_training=True,
        share_critic_params=True,
        enable_communication=False,  # Can enable for more complex coordination
        reward_sharing="mixed",
        reward_sharing_weight=0.7,

        # PPO settings
        learning_rate=3e-4,
        n_epochs=10,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,

        # Training settings
        normalize_advantages=True,
        normalize_rewards=True,

        # Hardware
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=4,

        # Logging
        log_interval=10,
        save_interval=50,
        experiment_name="cooperative_navigation"
    )

    # Create environments
    envs = [CooperativeNavigationEnv(n_agents=n_agents) for _ in range(n_envs)]

    # Create network configurations
    actor_config = NetworkConfig(
        layer_sizes=[64, 64],
        activation="tanh",
        use_layer_norm=False
    )

    critic_config = MultiAgentNetworkConfig(
        layer_sizes=[128, 128],
        activation="relu",
        use_layer_norm=True,
        n_agents=n_agents,
        use_agent_embeddings=True
    )

    # Create agents
    agents = []
    for i in range(n_agents):
        agent = MultiAgentPPOAgent(
            agent_id=i,
            n_agents=n_agents,
            observation_dim=envs[0].observation_space.shape[0],
            action_dim=envs[0].action_space.shape[0],
            actor_config=actor_config,
            critic_config=critic_config,
            continuous_actions=True,
            device=config.device,
            seed=42 + i
        )
        agents.append(agent)

    print(f"Created {n_agents} agents with centralized critic")
    print(f"Environment: Cooperative Navigation")
    print(f"Observation space: {envs[0].observation_space}")
    print(f"Action space: {envs[0].action_space}")

    # Create trainer
    trainer = MultiAgentPPOTrainer(
        agents=agents,
        envs=envs,
        config=config,
        experiment_name="cooperative_navigation",
        use_tensorboard=True,
        use_wandb=False
    )

    # Train
    print("\nStarting training...")
    trainer.train(
        total_timesteps=100000,
        rollout_length=2048
    )

    print("\nTraining completed!")

    # Test trained agents
    test_cooperative_agents(agents, CooperativeNavigationEnv(n_agents=n_agents))


def train_competitive_agents():
    """Train agents in a competitive scenario using wrapped single-agent envs."""
    print("\n=== Training Competitive Multi-Agent System ===\n")

    # Configuration
    n_agents = 2
    n_envs = 8

    # Create config
    config = MultiAgentConfig(
        # Multi-agent settings
        n_agents=n_agents,
        centralized_training=True,
        share_critic_params=False,  # Separate critics for competing agents
        share_critic_optimizer=False,
        reward_sharing="individual",  # Pure competition

        # PPO settings
        learning_rate=1e-4,  # Lower for competitive scenarios
        n_epochs=5,
        batch_size=512,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.1,  # Smaller clipping
        entropy_coef=0.02,  # Higher exploration

        # Training settings
        normalize_advantages=True,
        normalize_rewards=False,  # Don't normalize in competitive settings
        target_kl=0.01,  # Stricter constraint

        # Hardware
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=8,

        # Logging
        log_interval=5,
        save_interval=50,
        experiment_name="competitive_cartpole"
    )

    # Create wrapped environments (agents compete on same task)
    envs = [MultiAgentEnvWrapper("CartPole-v1", n_agents) for _ in range(n_envs)]

    # Create network configurations
    actor_config = NetworkConfig(
        layer_sizes=[128, 128],
        activation="tanh"
    )

    critic_config = MultiAgentNetworkConfig(
        layer_sizes=[256, 256],
        activation="relu",
        n_agents=n_agents,
        use_agent_embeddings=True
    )

    # Create agents
    agents = []
    for i in range(n_agents):
        agent = MultiAgentPPOAgent(
            agent_id=i,
            n_agents=n_agents,
            observation_dim=envs[0].observation_space.shape[0],
            action_dim=envs[0].action_space.n,
            actor_config=actor_config,
            critic_config=critic_config,
            continuous_actions=False,
            device=config.device,
            seed=42 + i
        )
        agents.append(agent)

    print(f"Created {n_agents} competitive agents")
    print(f"Environment: Competitive CartPole")

    # Create trainer
    trainer = MultiAgentPPOTrainer(
        agents=agents,
        envs=envs,
        config=config,
        experiment_name="competitive_cartpole",
        use_tensorboard=True,
        use_wandb=False
    )

    # Train
    print("\nStarting training...")
    trainer.train(
        total_timesteps=50000,
        rollout_length=1024
    )

    print("\nTraining completed!")


def test_cooperative_agents(agents: List[MultiAgentPPOAgent], env: Any, num_episodes: int = 5):
    """Test trained cooperative agents."""
    print("\n=== Testing Trained Cooperative Agents ===\n")

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_rewards = np.zeros(len(agents))
        steps = 0

        print(f"\nEpisode {episode + 1}:")

        while not done and steps < 100:
            # Get actions from all agents
            actions = []
            with torch.no_grad():
                for i, agent in enumerate(agents):
                    obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0)
                    action, _, _ = agent.act(obs_tensor, deterministic=True)
                    actions.append(action.cpu().numpy()[0])

            actions = np.array(actions)

            # Step environment
            obs, rewards, dones, truncated, info = env.step(actions)
            total_rewards += rewards
            steps += 1

            # Render periodically
            if steps % 20 == 0:
                env.render()

            # Check if all done
            done = all(dones) or truncated

        print(f"\nEpisode {episode + 1} finished:")
        print(f"  Steps: {steps}")
        print(f"  Rewards: {total_rewards}")
        print(f"  Success: {all(dones)}")


def main():
    """Run multi-agent training examples."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent PPO Training Examples")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["cooperative", "competitive", "both"],
        default="cooperative",
        help="Which scenario to run"
    )
    parser.add_argument(
        "--create-configs",
        action="store_true",
        help="Create example configuration files"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create example configs if requested
    if args.create_configs:
        create_example_configs()
        print("Example configurations created in configs/multi_agent/")
        return

    # Run training
    if args.scenario in ["cooperative", "both"]:
        train_cooperative_agents()

    if args.scenario in ["competitive", "both"]:
        train_competitive_agents()

    print("\n=== All training completed! ===")


if __name__ == "__main__":
    main()
