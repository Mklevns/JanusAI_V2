# janus/training/ppo/multi_agent_buffer.py
"""
Multi-Agent Rollout Buffer for storing experiences from multiple agents.
Supports both individual agent data and joint observations/actions for centralized training.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional, List
import threading
import logging

logger = logging.getLogger(__name__)


class MultiAgentRolloutBuffer:
    """
    Thread-safe rollout buffer for multi-agent PPO with centralized critic.

    Stores:
    - Individual observations, actions, rewards, dones for each agent
    - Joint observations and actions for centralized critic training
    - Values and log probabilities
    """

    def __init__(
        self,
        buffer_size: int,
        n_agents: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: torch.device,
        n_envs: int = 1,
    ):
        """
        Initialize multi-agent buffer.

        Args:
            buffer_size: Number of transitions to store
            n_agents: Number of agents
            obs_shape: Shape of individual agent observation
            action_shape: Shape of individual agent action
            device: Device for tensor operations
            n_envs: Number of parallel environments
        """
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")
        if n_agents <= 0:
            raise ValueError(f"n_agents must be positive, got {n_agents}")
        if n_envs <= 0:
            raise ValueError(f"n_envs must be positive, got {n_envs}")

        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.n_envs = n_envs
        self.ptr = 0
        self.lock = threading.Lock()
        self.full = False

        # Calculate joint dimensions
        self.joint_obs_shape = (n_agents * obs_shape[0],) + obs_shape[1:]
        if action_shape == ():  # Discrete actions
            self.joint_action_shape = (n_agents,)
        else:  # Continuous actions
            self.joint_action_shape = (n_agents * action_shape[0],) + action_shape[1:]

        # Pre-allocate arrays for individual agent data
        # Shape: [buffer_size, n_envs, n_agents, ...]
        self.observations = np.zeros(
            (buffer_size, n_envs, n_agents, *obs_shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (buffer_size, n_envs, n_agents, *action_shape), dtype=np.float32
        )
        self.rewards = np.zeros(
            (buffer_size, n_envs, n_agents), dtype=np.float32
        )
        self.dones = np.zeros(
            (buffer_size, n_envs, n_agents), dtype=np.float32
        )
        self.values = np.zeros(
            (buffer_size, n_envs, n_agents), dtype=np.float32
        )
        self.log_probs = np.zeros(
            (buffer_size, n_envs, n_agents), dtype=np.float32
        )

        # Pre-allocate arrays for joint data (used by centralized critic)
        self.joint_observations = np.zeros(
            (buffer_size, n_envs, *self.joint_obs_shape), dtype=np.float32
        )
        self.joint_actions = np.zeros(
            (buffer_size, n_envs, *self.joint_action_shape), dtype=np.float32
        )

        # Agent IDs for embeddings (constant across timesteps)
        self.agent_ids = np.arange(n_agents).reshape(1, 1, n_agents)
        self.agent_ids = np.tile(self.agent_ids, (buffer_size, n_envs, 1))

        logger.info(
            f"MultiAgentRolloutBuffer initialized: "
            f"buffer_size={buffer_size}, n_agents={n_agents}, "
            f"n_envs={n_envs}, device={device}"
        )

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
        joint_obs: Optional[np.ndarray] = None,
        joint_action: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add transitions from all agents.

        Args:
            obs: Individual observations [n_envs, n_agents, *obs_shape]
            action: Individual actions [n_envs, n_agents, *action_shape]
            reward: Rewards [n_envs, n_agents]
            done: Done flags [n_envs, n_agents]
            value: Value estimates [n_envs, n_agents]
            log_prob: Log probabilities [n_envs, n_agents]
            joint_obs: Joint observations [n_envs, *joint_obs_shape] (optional)
            joint_action: Joint actions [n_envs, *joint_action_shape] (optional)
        """
        with self.lock:
            if self.ptr >= self.buffer_size:
                logger.warning("Buffer overflow. Wrapping around.")
                self.ptr = 0
                self.full = True

            # Validate shapes
            expected_shapes = {
                "obs": (self.n_envs, self.n_agents, *self.obs_shape),
                "action": (self.n_envs, self.n_agents, *self.action_shape),
                "reward": (self.n_envs, self.n_agents),
                "done": (self.n_envs, self.n_agents),
                "value": (self.n_envs, self.n_agents),
                "log_prob": (self.n_envs, self.n_agents),
            }

            for name, (arr, expected) in zip(
                ["obs", "action", "reward", "done", "value", "log_prob"],
                [(obs, expected_shapes["obs"]),
                 (action, expected_shapes["action"]),
                 (reward, expected_shapes["reward"]),
                 (done, expected_shapes["done"]),
                 (value, expected_shapes["value"]),
                 (log_prob, expected_shapes["log_prob"])]
            ):
                if arr.shape != expected:
                    raise ValueError(
                        f"{name} shape mismatch: expected {expected}, got {arr.shape}"
                    )

            # Store individual agent data
            self.observations[self.ptr] = obs
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = done
            self.values[self.ptr] = value
            self.log_probs[self.ptr] = log_prob

            # Store or compute joint data
            if joint_obs is not None:
                self.joint_observations[self.ptr] = joint_obs
            else:
                # Flatten agent observations to create joint observation
                self.joint_observations[self.ptr] = obs.reshape(self.n_envs, -1)

            if joint_action is not None:
                self.joint_actions[self.ptr] = joint_action
            else:
                # Flatten agent actions to create joint action
                if self.action_shape == ():  # Discrete
                    self.joint_actions[self.ptr] = action.reshape(self.n_envs, -1)
                else:  # Continuous
                    self.joint_actions[self.ptr] = action.reshape(self.n_envs, -1)

            self.ptr += 1

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve all data as tensors for training.

        Returns:
            Dictionary containing all buffer data as torch tensors
        """
        with self.lock:
            if self.ptr == 0 and not self.full:
                raise RuntimeError("Buffer is empty")

            actual_size = self.buffer_size if self.full else self.ptr

            # Convert to tensors
            data = {
                # Individual agent data
                "observations": torch.as_tensor(
                    self.observations[:actual_size], device=self.device
                ),
                "actions": torch.as_tensor(
                    self.actions[:actual_size], device=self.device
                ),
                "rewards": torch.as_tensor(
                    self.rewards[:actual_size], device=self.device
                ),
                "dones": torch.as_tensor(
                    self.dones[:actual_size], device=self.device
                ),
                "values": torch.as_tensor(
                    self.values[:actual_size], device=self.device
                ),
                "log_probs": torch.as_tensor(
                    self.log_probs[:actual_size], device=self.device
                ),
                # Joint data for centralized critic
                "joint_observations": torch.as_tensor(
                    self.joint_observations[:actual_size], device=self.device
                ),
                "joint_actions": torch.as_tensor(
                    self.joint_actions[:actual_size], device=self.device
                ),
                # Agent IDs for embeddings
                "agent_ids": torch.as_tensor(
                    self.agent_ids[:actual_size], device=self.device
                ),
            }

            return data

    def get_agent_data(self, agent_id: int) -> Dict[str, torch.Tensor]:
        """
        Get data for a specific agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Dictionary containing the specified agent's data
        """
        if agent_id < 0 or agent_id >= self.n_agents:
            raise ValueError(f"Invalid agent_id: {agent_id}")

        all_data = self.get()

        # Extract specific agent's data
        agent_data = {
            "observations": all_data["observations"][:, :, agent_id],
            "actions": all_data["actions"][:, :, agent_id],
            "rewards": all_data["rewards"][:, :, agent_id],
            "dones": all_data["dones"][:, :, agent_id],
            "values": all_data["values"][:, :, agent_id],
            "log_probs": all_data["log_probs"][:, :, agent_id],
            # Joint data remains the same for all agents
            "joint_observations": all_data["joint_observations"],
            "joint_actions": all_data["joint_actions"],
            "agent_ids": all_data["agent_ids"],
        }

        return agent_data

    def clear(self) -> None:
        """Reset the buffer for next rollout."""
        with self.lock:
            self.ptr = 0
            self.full = False
            logger.debug("MultiAgentRolloutBuffer cleared.")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get buffer statistics.

        Returns:
            Dictionary with buffer statistics
        """
        with self.lock:
            actual_size = self.buffer_size if self.full else self.ptr

            if actual_size == 0:
                return {
                    "buffer_size": 0,
                    "mean_rewards": np.zeros(self.n_agents),
                    "std_rewards": np.zeros(self.n_agents),
                    "mean_values": np.zeros(self.n_agents),
                }

            # Calculate per-agent statistics
            mean_rewards = self.rewards[:actual_size].mean(axis=(0, 1))
            std_rewards = self.rewards[:actual_size].std(axis=(0, 1))
            mean_values = self.values[:actual_size].mean(axis=(0, 1))

            return {
                "buffer_size": actual_size,
                "mean_rewards": mean_rewards,
                "std_rewards": std_rewards,
                "mean_values": mean_values,
                "total_rewards": self.rewards[:actual_size].sum(axis=0).mean(axis=0),
            }


# Demonstration and testing
if __name__ == '__main__':
    """Test multi-agent rollout buffer functionality."""

    print("=== Multi-Agent Rollout Buffer Demo ===\n")

    # Configuration
    buffer_size = 100
    n_agents = 3
    n_envs = 4
    obs_shape = (8,)
    action_shape = ()  # Discrete actions
    device = torch.device("cpu")

    # Create buffer
    buffer = MultiAgentRolloutBuffer(
        buffer_size=buffer_size,
        n_agents=n_agents,
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        n_envs=n_envs
    )

    print(f"Created buffer: size={buffer_size}, agents={n_agents}, envs={n_envs}")
    print(f"Joint observation shape: {buffer.joint_obs_shape}")
    print(f"Joint action shape: {buffer.joint_action_shape}\n")

    # Test adding data
    print("Adding sample data...")
    for step in range(10):
        # Generate random data
        obs = np.random.randn(n_envs, n_agents, *obs_shape).astype(np.float32)
        actions = np.random.randint(0, 4, size=(n_envs, n_agents)).astype(np.float32)
        rewards = np.random.randn(n_envs, n_agents).astype(np.float32)
        dones = np.random.rand(n_envs, n_agents) > 0.9
        values = np.random.randn(n_envs, n_agents).astype(np.float32)
        log_probs = np.random.randn(n_envs, n_agents).astype(np.float32) - 1

        buffer.add(
            obs=obs,
            action=actions,
            reward=rewards,
            done=dones.astype(np.float32),
            value=values,
            log_prob=log_probs
        )

    print(f"✓ Added {10} steps to buffer\n")

    # Test retrieval
    print("Retrieving data...")
    data = buffer.get()

    print("Data shapes:")
    for key, tensor in data.items():
        print(f"  {key}: {tensor.shape}")

    # Test agent-specific data
    print("\nGetting data for agent 0...")
    agent_data = buffer.get_agent_data(0)
    print("Agent 0 observation shape:", agent_data["observations"].shape)
    print("Agent 0 rewards shape:", agent_data["rewards"].shape)

    # Test statistics
    print("\nBuffer statistics:")
    stats = buffer.get_statistics()
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    # Test thread safety
    print("\nTesting thread safety...")
    import threading

    def add_data_thread(thread_id):
        for _ in range(20):
            obs = np.random.randn(n_envs, n_agents, *obs_shape).astype(np.float32)
            actions = np.random.randint(0, 4, size=(n_envs, n_agents)).astype(np.float32)
            rewards = np.ones((n_envs, n_agents), dtype=np.float32) * thread_id
            dones = np.zeros((n_envs, n_agents), dtype=np.float32)
            values = np.ones((n_envs, n_agents), dtype=np.float32) * 0.5
            log_probs = np.ones((n_envs, n_agents), dtype=np.float32) * -0.5

            buffer.add(obs, actions, rewards, dones, values, log_probs)

    threads = []
    for i in range(4):
        t = threading.Thread(target=add_data_thread, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("✓ Thread safety test completed")
    print(f"Final buffer size: {buffer.ptr}")

    print("\n=== Demo completed successfully! ===")
