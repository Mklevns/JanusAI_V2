# janus/agents/ppo_agent.py

"""
A production-hardened PPOAgent that is configurable, robust, and extensible.
"""
import logging
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.distributions import Categorical
from dataclasses import dataclass

from janus.core.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Suggestion 2: Parameterize Network Architecture via a Configurable Schema
@dataclass
class NetworkConfig:
    """A typed configuration schema for building neural networks."""
    layer_sizes: List[int]
    activation: str = "tanh"  # e.g., "tanh", "relu"
    use_layer_norm: bool = False

    def get_activation(self) -> nn.Module:
        """Returns the PyTorch activation function module."""
        if self.activation == "tanh":
            return nn.Tanh()
        elif self.activation == "relu":
            return nn.ReLU()
        raise ValueError(f"Unsupported activation function: {self.activation}")

def build_network(input_dim: int, output_dim: int, config: NetworkConfig) -> nn.Sequential:
    """Builds a neural network from a NetworkConfig."""
    layers = []
    layer_dims = [input_dim] + config.layer_sizes
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        if config.use_layer_norm:
            layers.append(nn.LayerNorm(layer_dims[i+1]))
        layers.append(config.get_activation())
    layers.append(nn.Linear(layer_dims[-1], output_dim))
    return nn.Sequential(*layers)


class PPOAgent(BaseAgent, nn.Module):
    """
    A robust, configurable SolverAgent that uses Proximal Policy Optimization.

    This agent's architecture is defined by a configuration object, supports
    action masking for environments with dynamic action spaces, and includes
    methods for checkpointing, making it suitable for production-scale
    research within the JanusAI V2 framework.
    """
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        actor_config: NetworkConfig,
        critic_config: NetworkConfig,
        device: str = 'cpu'
    ):
        """
        Initializes the PPOAgent.

        Args:
            observation_dim (int): The dimensionality of the observation space.
            action_dim (int): The dimensionality of the action space.
            actor_config (NetworkConfig): Configuration for the actor network.
            critic_config (NetworkConfig): Configuration for the critic network.
            device (str): The device (e.g., 'cpu', 'cuda') to run the agent on.
        """
        # Suggestion 3: Robust cooperative multiple inheritance
        super().__init__()
        
        self.device = torch.device(device)
        self.action_dim = action_dim

        # Build actor and critic networks from configs
        self.actor = build_network(observation_dim, action_dim, actor_config)
        self.critic = build_network(observation_dim, 1, critic_config)
        
        # Suggestion 5: Explicitly move networks to the specified device
        self.to(self.device)
        logger.info(f"PPOAgent initialized on device '{self.device}'")

    def act(
        self,
        observation: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects an action based on the policy for inference, with optional masking.

        Args:
            observation: The current state observation tensor.
            action_mask: An optional boolean tensor where `True` indicates a
                         valid action. Invalid actions will not be chosen.

        Returns:
            A tuple containing the sampled action and its log probability.
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        
        # Suggestion 5: Ensure tensors are on the correct device
        obs_tensor = observation.to(self.device)
        
        action_logits = self.actor(obs_tensor)

        # Suggestion 7: Support for Action Masking
        if action_mask is not None:
            mask_tensor = action_mask.to(self.device)
            action_logits[~mask_tensor] = -1e9  # Set logits of invalid actions to a large negative number

        action_probs = nn.functional.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def evaluate(
        self,
        observation: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates a state-action pair during training."""
        obs_tensor = observation.to(self.device)
        action_tensor = action.to(self.device)
        
        action_probs = self.actor(obs_tensor)
        dist = Categorical(action_probs)
        action_logprob = dist.log_prob(action_tensor)
        dist_entropy = dist.entropy()
        state_value = self.critic(obs_tensor)

        return action_logprob, state_value, dist_entropy

    # Suggestion 10: Checkpointing & Serialization
    def save_checkpoint(self, path: str) -> None:
        """Saves the agent's state to a file."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, path)
        logger.info(f"Agent checkpoint saved to '{path}'")

    def load_checkpoint(self, path: str) -> None:
        """Loads the agent's state from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        logger.info(f"Agent checkpoint loaded from '{path}'")
        self.to(self.device) # Ensure model is on the correct device after loading
        
    def learn(self, *args, **kwargs) -> None:
        # Suggestion 4: Raise error to enforce trainer-based learning
        raise NotImplementedError("PPOAgent learning must be handled by a dedicated PPOTrainer.")

if __name__ == '__main__':
    """Demonstrates the enhanced, hardened PPOAgent."""
    logging.basicConfig(level=logging.INFO)
    logger.info("--- Demonstrating Hardened PPOAgent ---")

    # 1. Define network architecture via config objects
    actor_conf = NetworkConfig(layer_sizes=[128, 128], activation="relu")
    critic_conf = NetworkConfig(layer_sizes=[64, 64], activation="tanh", use_layer_norm=True)

    # 2. Instantiate the agent
    agent = PPOAgent(
        observation_dim=10,
        action_dim=4,
        actor_config=actor_conf,
        critic_config=critic_conf,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("\nActor Architecture:\n", agent.actor)
    print("\nCritic Architecture:\n", agent.critic)
    
    # 3. Demonstrate action masking
    logger.info("\n--- Demonstrating Action Masking ---")
    dummy_obs = torch.randn(1, 10)
    # Mask to only allow actions 1 and 3
    action_mask = torch.tensor([False, True, False, True])
    
    for _ in range(10):
        action, _ = agent.act(dummy_obs, action_mask=action_mask)
        assert action.item() in [1, 3], f"Action masking failed! Got action {action.item()}"
    logger.info("Action masking successful: all sampled actions were valid.")

    # 4. Demonstrate checkpointing
    logger.info("\n--- Demonstrating Checkpointing ---")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "ppo_agent.pt")
        agent.save_checkpoint(checkpoint_path)
        
        # Create a new agent and load the checkpoint
        new_agent = PPOAgent(10, 4, actor_conf, critic_conf)
        new_agent.load_checkpoint(checkpoint_path)
        logger.info("Successfully loaded checkpoint into a new agent instance.")

        # Verify that the weights are the same
        assert torch.equal(
            agent.actor[0].weight, new_agent.actor[0].weight
        ), "Actor weights do not match after loading checkpoint!"
        logger.info("Weight verification successful.")
