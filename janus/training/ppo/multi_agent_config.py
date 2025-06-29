# janus/training/ppo/multi_agent_config.py
"""
Extended configuration for Multi-Agent PPO training.
Adds multi-agent specific parameters to the base PPOConfig.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import yaml

from janus.training.ppo.config import PPOConfig


@dataclass
class MultiAgentConfig(PPOConfig):
    """
    Extended configuration for Multi-Agent PPO.

    Inherits all base PPO parameters and adds multi-agent specific settings.
    """

    # Multi-agent specific parameters
    n_agents: int = field(
        default=2,
        metadata={"description": "Number of agents in the system"}
    )

    share_actor_params: bool = field(
        default=False,
        metadata={"description": "Whether agents share actor network parameters"}
    )

    share_critic_params: bool = field(
        default=True,
        metadata={"description": "Whether agents share critic network parameters"}
    )

    share_critic_optimizer: bool = field(
        default=True,
        metadata={"description": "Whether to use single optimizer for all critics"}
    )

    agent_id_embedding_dim: int = field(
        default=16,
        metadata={"description": "Dimension of agent ID embeddings"}
    )

    use_agent_embeddings: bool = field(
        default=True,
        metadata={"description": "Whether to use agent ID embeddings in critic"}
    )

    # Communication settings
    enable_communication: bool = field(
        default=False,
        metadata={"description": "Enable inter-agent communication"}
    )

    communication_dim: int = field(
        default=32,
        metadata={"description": "Dimension of communication messages"}
    )

    communication_rounds: int = field(
        default=1,
        metadata={"description": "Number of communication rounds per step"}
    )

    # Reward settings
    reward_sharing: str = field(
        default="individual",
        metadata={
            "description": "Reward sharing mode: 'individual', 'shared', 'mixed'",
            "choices": ["individual", "shared", "mixed"]
        }
    )

    reward_sharing_weight: float = field(
        default=0.5,
        metadata={"description": "Weight for shared rewards in 'mixed' mode"}
    )

    # Training settings
    centralized_training: bool = field(
        default=True,
        metadata={"description": "Use centralized training with decentralized execution"}
    )

    independent_ppo: bool = field(
        default=False,
        metadata={"description": "Train agents independently (no centralized critic)"}
    )

    # Environment settings
    env_type: str = field(
        default="multi_agent",
        metadata={
            "description": "Environment type: 'multi_agent' or 'single_agent'",
            "choices": ["multi_agent", "single_agent"]
        }
    )

    # Agent grouping (for competitive/cooperative scenarios)
    agent_groups: Optional[Dict[str, List[int]]] = field(
        default=None,
        metadata={
            "description": "Group agents for team-based scenarios",
            "example": {"team_1": [0, 1], "team_2": [2, 3]}
        }
    )

    # Logging
    log_agent_metrics: bool = field(
        default=True,
        metadata={"description": "Log individual agent metrics"}
    )

    log_communication_stats: bool = field(
        default=True,
        metadata={"description": "Log communication statistics if enabled"}
    )

    @classmethod
    def from_yaml(cls, path: Path) -> "MultiAgentConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Handle nested dictionaries properly
        if 'experiment_tags' in data and data['experiment_tags'] is None:
            data['experiment_tags'] = {}
        if 'agent_groups' in data and data['agent_groups'] is None:
            data['agent_groups'] = None

        return cls(**data)

    def validate(self) -> None:
        """Validate configuration parameters."""
        # Call parent validation
        super().validate()

        # Multi-agent specific validation
        if self.n_agents < 1:
            raise ValueError(f"n_agents must be positive, got {self.n_agents}")

        if self.reward_sharing not in ["individual", "shared", "mixed"]:
            raise ValueError(
                f"reward_sharing must be 'individual', 'shared', or 'mixed', "
                f"got {self.reward_sharing}"
            )

        if self.reward_sharing == "mixed" and not (0 <= self.reward_sharing_weight <= 1):
            raise ValueError(
                f"reward_sharing_weight must be in [0, 1], "
                f"got {self.reward_sharing_weight}"
            )

        if self.independent_ppo and self.centralized_training:
            raise ValueError(
                "Cannot have both independent_ppo and centralized_training enabled"
            )

        if self.enable_communication and self.communication_dim < 1:
            raise ValueError(
                f"communication_dim must be positive when communication is enabled, "
                f"got {self.communication_dim}"
            )

        if self.agent_groups is not None:
            all_agents = set()
            for group_name, agent_ids in self.agent_groups.items():
                for agent_id in agent_ids:
                    if agent_id < 0 or agent_id >= self.n_agents:
                        raise ValueError(
                            f"Invalid agent ID {agent_id} in group '{group_name}'. "
                            f"Must be in [0, {self.n_agents-1}]"
                        )
                    if agent_id in all_agents:
                        raise ValueError(
                            f"Agent {agent_id} appears in multiple groups"
                        )
                    all_agents.add(agent_id)


# Example configuration files
EXAMPLE_COOPERATIVE_CONFIG = """
# Multi-Agent PPO Configuration - Cooperative Setting
# All agents work together towards a common goal

# Base PPO parameters
learning_rate: 0.0003
gamma: 0.99
gae_lambda: 0.95
clip_epsilon: 0.2
value_coef: 0.5
entropy_coef: 0.01
n_epochs: 10
batch_size: 256

# Multi-agent parameters
n_agents: 4
share_actor_params: false  # Each agent has its own policy
share_critic_params: true   # Shared critic for stability
centralized_training: true  # MADDPG-style training

# Reward configuration
reward_sharing: "mixed"     # Agents get both individual and team rewards
reward_sharing_weight: 0.7  # 70% weight on shared rewards

# Communication
enable_communication: true
communication_dim: 64
communication_rounds: 2

# Training settings
normalize_advantages: true
normalize_rewards: true
max_grad_norm: 0.5
use_mixed_precision: true

# Scheduling
lr_schedule: "cosine"
lr_end: 0.00001
entropy_schedule: "linear"
entropy_end: 0.001

# Hardware
device: "cuda"
num_workers: 8

# Logging
log_interval: 10
save_interval: 100
experiment_name: "multi_agent_cooperative"
"""

EXAMPLE_COMPETITIVE_CONFIG = """
# Multi-Agent PPO Configuration - Competitive Setting
# Agents compete in teams

# Base PPO parameters
learning_rate: 0.0001       # Lower LR for competitive scenarios
gamma: 0.99
gae_lambda: 0.95
clip_epsilon: 0.1           # Smaller clipping for stability
value_coef: 0.5
entropy_coef: 0.02          # Higher entropy for exploration
n_epochs: 5                 # Fewer epochs to prevent overfitting
batch_size: 512

# Multi-agent parameters
n_agents: 4
share_actor_params: false
share_critic_params: false  # Separate critics for competing teams
centralized_training: true

# Team configuration
agent_groups:
  team_red: [0, 1]
  team_blue: [2, 3]

# Reward configuration
reward_sharing: "individual"  # Competitive - individual rewards only

# No communication between competing teams
enable_communication: false

# Training settings
normalize_advantages: true
normalize_rewards: false    # Don't normalize in competitive settings
max_grad_norm: 0.5
target_kl: 0.01            # Stricter KL constraint

# Hardware
device: "cuda"
num_workers: 16

# Logging
log_interval: 5
save_interval: 50
log_agent_metrics: true
experiment_name: "multi_agent_competitive"
"""

EXAMPLE_MIXED_CONFIG = """
# Multi-Agent PPO Configuration - Mixed Cooperative-Competitive
# Some agents cooperate, others compete

# Base PPO parameters
learning_rate: 0.0002
gamma: 0.995               # Longer horizon
gae_lambda: 0.98
clip_epsilon: 0.15
value_coef: 0.5
entropy_coef: 0.015
n_epochs: 8
batch_size: 384

# Multi-agent parameters
n_agents: 6
share_actor_params: false
share_critic_params: false
share_critic_optimizer: false  # Separate optimizers for different teams

# Mixed teams - cooperation within, competition between
agent_groups:
  predators: [0, 1, 2]     # Cooperate to catch prey
  prey: [3, 4, 5]          # Cooperate to evade

# Reward configuration
reward_sharing: "mixed"
reward_sharing_weight: 0.5  # Balance individual and team performance

# Communication within teams only
enable_communication: true
communication_dim: 32
communication_rounds: 1

# Training settings
normalize_advantages: true
normalize_rewards: true
max_grad_norm: 1.0
use_mixed_precision: true

# Adaptive scheduling
lr_schedule: "cosine"
lr_end: 0.00005
clip_schedule: "linear"
clip_end: 0.05
entropy_schedule: "cosine"
entropy_end: 0.005
target_kl: 0.02

# Hardware
device: "cuda"
num_workers: 12
use_subprocess_envs: true

# Logging
log_interval: 10
save_interval: 100
eval_interval: 500
log_agent_metrics: true
log_communication_stats: true
experiment_name: "predator_prey_multi_agent"
experiment_tags:
  scenario: "mixed_coop_comp"
  n_predators: 3
  n_prey: 3
"""


def create_example_configs(output_dir: Path = Path("configs/multi_agent")):
    """Create example configuration files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save cooperative config
    with open(output_dir / "cooperative.yaml", 'w') as f:
        f.write(EXAMPLE_COOPERATIVE_CONFIG)

    # Save competitive config
    with open(output_dir / "competitive.yaml", 'w') as f:
        f.write(EXAMPLE_COMPETITIVE_CONFIG)

    # Save mixed config
    with open(output_dir / "mixed.yaml", 'w') as f:
        f.write(EXAMPLE_MIXED_CONFIG)

    print(f"Created example configs in {output_dir}")


if __name__ == "__main__":
    """Demonstrate configuration usage."""
    import tempfile

    print("=== Multi-Agent Configuration Demo ===\n")

    # Create example configs
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        create_example_configs(config_dir)

        # Load and validate cooperative config
        print("Loading cooperative configuration...")
        coop_config = MultiAgentConfig.from_yaml(config_dir / "cooperative.yaml")
        coop_config.validate()
        print(f"✓ Cooperative config: {coop_config.n_agents} agents, "
              f"communication={'enabled' if coop_config.enable_communication else 'disabled'}")

        # Load and validate competitive config
        print("\nLoading competitive configuration...")
        comp_config = MultiAgentConfig.from_yaml(config_dir / "competitive.yaml")
        comp_config.validate()
        print(f"✓ Competitive config: {comp_config.n_agents} agents in "
              f"{len(comp_config.agent_groups)} teams")

        # Load and validate mixed config
        print("\nLoading mixed configuration...")
        mixed_config = MultiAgentConfig.from_yaml(config_dir / "mixed.yaml")
        mixed_config.validate()
        print(f"✓ Mixed config: {mixed_config.n_agents} agents, "
              f"reward sharing mode: {mixed_config.reward_sharing}")

    print("\n=== Configuration demo completed! ===")
