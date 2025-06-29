# janus/agents/multi_agent_ppo.py
"""
Multi-Agent PPO with centralized critic for training and decentralized execution.
Based on MADDPG principles adapted for PPO.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
from dataclasses import dataclass

from janus.agents.ppo_agent import PPOAgent, NetworkConfig, build_network

logger = logging.getLogger(__name__)


@dataclass
class MultiAgentNetworkConfig(NetworkConfig):
    """Extended network configuration for multi-agent settings."""
    n_agents: int = 1
    agent_id_embedding_dim: int = 16  # For agent identification
    use_agent_embeddings: bool = True
    share_actor_params: bool = False  # Whether agents share actor parameters
    share_critic_params: bool = True  # Whether agents share critic parameters


class CentralizedCritic(nn.Module):
    """
    Centralized critic that observes all agents' states and actions.
    During training, it receives the full joint observation-action space.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim_per_agent: int,
        action_dim_per_agent: int,
        config: MultiAgentNetworkConfig,
        continuous_actions: bool = False
    ):
        super().__init__()

        self.n_agents = n_agents
        self.obs_dim_per_agent = obs_dim_per_agent
        self.action_dim_per_agent = action_dim_per_agent
        self.continuous_actions = continuous_actions

        # Calculate input dimension for centralized critic
        # It sees all agents' observations and actions
        joint_obs_dim = n_agents * obs_dim_per_agent
        joint_action_dim = n_agents * action_dim_per_agent

        # For discrete actions, we'll use one-hot encoding
        if not continuous_actions:
            joint_action_dim = n_agents * action_dim_per_agent

        total_input_dim = joint_obs_dim + joint_action_dim

        # Add agent embeddings if enabled
        if config.use_agent_embeddings:
            self.agent_embeddings = nn.Embedding(n_agents, config.agent_id_embedding_dim)
            total_input_dim += n_agents * config.agent_id_embedding_dim
        else:
            self.agent_embeddings = None

        # Build the critic network
        self.network = build_network(total_input_dim, n_agents, config)

    def forward(
        self,
        joint_observations: torch.Tensor,
        joint_actions: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of centralized critic.

        Args:
            joint_observations: [batch_size, n_agents * obs_dim]
            joint_actions: [batch_size, n_agents * action_dim] (one-hot for discrete)
            agent_ids: [batch_size, n_agents] agent indices for embeddings

        Returns:
            values: [batch_size, n_agents] value estimates for each agent
        """
        batch_size = joint_observations.shape[0]

        # Prepare joint actions (one-hot encode if discrete)
        if not self.continuous_actions and joint_actions.dim() == 2:
            # Reshape to [batch_size, n_agents]
            agent_actions = joint_actions.view(batch_size, self.n_agents)
            # One-hot encode
            joint_actions_onehot = torch.zeros(
                batch_size, self.n_agents * self.action_dim_per_agent,
                device=joint_actions.device
            )
            for i in range(self.n_agents):
                agent_action = agent_actions[:, i].long()
                start_idx = i * self.action_dim_per_agent
                joint_actions_onehot.scatter_(
                    1,
                    start_idx + agent_action.unsqueeze(1),
                    1.0
                )
            joint_actions = joint_actions_onehot

        # Concatenate all inputs
        inputs = [joint_observations, joint_actions]

        # Add agent embeddings if enabled
        if self.agent_embeddings is not None and agent_ids is not None:
            agent_embeds = self.agent_embeddings(agent_ids)  # [batch_size, n_agents, embed_dim]
            agent_embeds = agent_embeds.view(batch_size, -1)  # Flatten
            inputs.append(agent_embeds)

        critic_input = torch.cat(inputs, dim=-1)
        values = self.network(critic_input)

        return values


class MultiAgentPPOAgent(PPOAgent):
    """
    Multi-Agent PPO with centralized training and decentralized execution.
    Each agent has its own actor (policy) but shares information through a centralized critic.
    """

    def __init__(
        self,
        agent_id: int,
        n_agents: int,
        observation_dim: int,
        action_dim: int,
        actor_config: NetworkConfig,
        critic_config: MultiAgentNetworkConfig,
        continuous_actions: bool = False,
        device: str = 'cpu',
        seed: Optional[int] = None
    ):
        """
        Initialize a multi-agent PPO agent.

        Args:
            agent_id: Unique identifier for this agent
            n_agents: Total number of agents in the system
            observation_dim: Dimension of each agent's observation
            action_dim: Dimension of each agent's action space
            actor_config: Configuration for the actor network
            critic_config: Configuration for the centralized critic
            continuous_actions: Whether actions are continuous
            device: Device to run on
            seed: Random seed
        """
        # Initialize parent PPOAgent (creates actor)
        super().__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
            actor_config=actor_config,
            critic_config=critic_config,  # This will be overridden
            continuous_actions=continuous_actions,
            device=device,
            seed=seed
        )

        self.agent_id = agent_id
        self.n_agents = n_agents

        # Replace the critic with a centralized one
        self.centralized_critic = CentralizedCritic(
            n_agents=n_agents,
            obs_dim_per_agent=observation_dim,
            action_dim_per_agent=action_dim,
            config=critic_config,
            continuous_actions=continuous_actions
        ).to(self.device)

        # Remove the individual critic to avoid confusion
        delattr(self, 'critic')

        logger.info(
            f"MultiAgentPPOAgent {agent_id} initialized with centralized critic "
            f"(n_agents={n_agents}, device={device})"
        )

    def evaluate_with_central_critic(
        self,
        local_observations: torch.Tensor,
        local_actions: torch.Tensor,
        joint_observations: torch.Tensor,
        joint_actions: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions using local actor and centralized critic.

        Args:
            local_observations: This agent's observations [batch_size, obs_dim]
            local_actions: This agent's actions [batch_size, action_dim]
            joint_observations: All agents' observations [batch_size, n_agents * obs_dim]
            joint_actions: All agents' actions [batch_size, n_agents * action_dim]
            agent_ids: Agent identifiers for embeddings [batch_size, n_agents]

        Returns:
            log_probs: Log probabilities of the actions
            values: Value estimates from centralized critic
            entropy: Entropy of the action distribution
        """
        # Move to device
        local_obs = local_observations.to(self.device)
        local_acts = local_actions.to(self.device)
        joint_obs = joint_observations.to(self.device)
        joint_acts = joint_actions.to(self.device)

        # Get action distribution from local actor
        if self.continuous_actions:
            mean = self.actor(local_obs)
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(local_acts).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            action_logits = self.actor(local_obs)
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(local_acts.squeeze(-1))
            entropy = dist.entropy()

        # Get values from centralized critic
        all_values = self.centralized_critic(joint_obs, joint_acts, agent_ids)

        # Extract this agent's value
        values = all_values[:, self.agent_id]

        return log_probs, values, entropy

    def get_value_centralized(
        self,
        joint_observations: torch.Tensor,
        joint_actions: Optional[torch.Tensor] = None,
        agent_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get value estimates from the centralized critic.

        Args:
            joint_observations: All agents' observations
            joint_actions: All agents' actions (can be zeros for value estimation)
            agent_ids: Agent identifiers

        Returns:
            values: Value estimates for all agents [batch_size, n_agents]
        """
        joint_obs = joint_observations.to(self.device)

        # If no actions provided, use zeros (for value bootstrapping)
        if joint_actions is None:
            batch_size = joint_obs.shape[0]
            if self.continuous_actions:
                joint_actions = torch.zeros(
                    batch_size, self.n_agents * self.action_dim,
                    device=self.device
                )
            else:
                # For discrete, we'll use the first action (index 0) as default
                joint_actions = torch.zeros(
                    batch_size, self.n_agents,
                    device=self.device
                )
        else:
            joint_actions = joint_actions.to(self.device)

        with torch.no_grad():
            values = self.centralized_critic(joint_obs, joint_actions, agent_ids)

        return values

    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save both actor and centralized critic."""
        checkpoint = {
            'agent_id': self.agent_id,
            'n_agents': self.n_agents,
            'actor_state_dict': self.actor.state_dict(),
            'centralized_critic_state_dict': self.centralized_critic.state_dict(),
        }
        if self.continuous_actions:
            checkpoint['log_std'] = self.log_std

        checkpoint.update(kwargs)
        torch.save(checkpoint, path)
        logger.info(f"Multi-agent checkpoint saved to '{path}'")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load both actor and centralized critic."""
        checkpoint = torch.load(path, map_location=self.device)

        # Verify agent configuration matches
        assert checkpoint['agent_id'] == self.agent_id, "Agent ID mismatch"
        assert checkpoint['n_agents'] == self.n_agents, "Number of agents mismatch"

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.centralized_critic.load_state_dict(checkpoint['centralized_critic_state_dict'])

        if self.continuous_actions and 'log_std' in checkpoint:
            self.log_std.data = checkpoint['log_std']

        logger.info(f"Multi-agent checkpoint loaded from '{path}'")
        return checkpoint


# Demonstration and testing
if __name__ == '__main__':
    """Demonstrate multi-agent PPO with centralized critic."""
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO)

    print("=== Multi-Agent PPO Demo ===\n")

    # Configuration
    n_agents = 3
    obs_dim = 8
    action_dim = 4
    batch_size = 32

    # Create network configs
    actor_config = NetworkConfig(layer_sizes=[64, 64], activation="tanh")
    critic_config = MultiAgentNetworkConfig(
        layer_sizes=[128, 128],
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
            observation_dim=obs_dim,
            action_dim=action_dim,
            actor_config=actor_config,
            critic_config=critic_config,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        agents.append(agent)

    print(f"Created {n_agents} agents with centralized critic\n")

    # Test forward pass
    print("Testing forward pass...")

    # Generate random data
    local_obs = [torch.randn(batch_size, obs_dim) for _ in range(n_agents)]
    joint_obs = torch.cat(local_obs, dim=1)  # [batch_size, n_agents * obs_dim]

    # Get actions from each agent (decentralized execution)
    actions = []
    log_probs = []

    for i, agent in enumerate(agents):
        action, log_prob, _ = agent.act(local_obs[i])
        actions.append(action)
        log_probs.append(log_prob)

    print(f"✓ Decentralized action selection completed")

    # Prepare joint actions
    if isinstance(actions[0], torch.Tensor) and actions[0].dim() == 1:
        joint_actions = torch.stack(actions, dim=1)  # [batch_size, n_agents]
    else:
        joint_actions = torch.cat(actions, dim=1)

    # Test centralized critic evaluation
    agent_ids = torch.arange(n_agents).unsqueeze(0).repeat(batch_size, 1)

    for i, agent in enumerate(agents):
        log_prob, value, entropy = agent.evaluate_with_central_critic(
            local_observations=local_obs[i],
            local_actions=actions[i] if actions[i].dim() > 1 else actions[i].unsqueeze(-1),
            joint_observations=joint_obs,
            joint_actions=joint_actions,
            agent_ids=agent_ids
        )

        print(f"Agent {i}: value={value.mean().item():.3f}, "
              f"entropy={entropy.mean().item():.3f}")

    print("\n✓ Centralized critic evaluation completed")

    # Test value estimation
    all_values = agents[0].get_value_centralized(joint_obs, joint_actions, agent_ids)
    print(f"\nCentralized value estimates shape: {all_values.shape}")
    print(f"Mean values per agent: {all_values.mean(dim=0).cpu().numpy()}")

    # Test checkpoint save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
        agents[0].save_checkpoint(tmp.name)
        agents[0].load_checkpoint(tmp.name)
        print("\n✓ Checkpoint save/load successful")

    print("\n=== Demo completed successfully! ===")
