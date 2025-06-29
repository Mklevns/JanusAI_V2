# Multi-Agent PPO Integration Guide

## Overview

This implementation brings MADDPG-style centralized training with decentralized execution to your JanusAI_V2 framework. The key innovation is that during training, each agent's critic can observe the states and actions of all agents, leading to more stable learning in multi-agent settings.

## Architecture Components

### 1. **MultiAgentPPOAgent** (`janus/agents/multi_agent_ppo.py`)
- Extends the base `PPOAgent` with a centralized critic
- Each agent maintains its own actor (policy) for decentralized execution
- The centralized critic observes joint states and actions during training
- Supports both discrete and continuous action spaces

### 2. **MultiAgentRolloutBuffer** (`janus/training/ppo/multi_agent_buffer.py`)
- Stores experiences from multiple agents simultaneously
- Maintains both individual agent data and joint observations/actions
- Thread-safe for parallel environment collection
- Efficient memory layout for fast training

### 3. **MultiAgentPPOTrainer** (`janus/training/ppo/multi_agent_trainer.py`)
- Orchestrates the training process for all agents
- Implements centralized value function updates
- Supports various reward sharing schemes
- Handles hyperparameter scheduling and logging

### 4. **MultiAgentConfig** (`janus/training/ppo/multi_agent_config.py`)
- Extends PPOConfig with multi-agent specific parameters
- Supports cooperative, competitive, and mixed scenarios
- Configurable communication and parameter sharing

## Key Features

### Centralized Training, Decentralized Execution (CTDE)
- **Training**: Critics observe all agents' observations and actions
- **Execution**: Each agent acts based only on its local observation
- **Benefit**: Addresses non-stationarity in multi-agent learning

### Flexible Agent Configurations
- **Independent Actors**: Each agent has its own policy network
- **Shared Critics**: Option to share critic parameters for cooperative scenarios
- **Agent Embeddings**: Learnable embeddings to help the critic distinguish agents

### Reward Sharing Options
- **Individual**: Each agent optimizes its own reward
- **Shared**: All agents receive the same team reward
- **Mixed**: Weighted combination of individual and shared rewards

## Integration Steps

### 1. **Install Dependencies**
No additional dependencies required beyond the existing PPO implementation.

### 2. **File Placement**
```
janus/
├── agents/
│   └── multi_agent_ppo.py          # Multi-agent PPO agent
├── training/
│   └── ppo/
│       ├── multi_agent_buffer.py   # Multi-agent rollout buffer
│       ├── multi_agent_trainer.py  # Multi-agent trainer
│       └── multi_agent_config.py   # Configuration extension
└── examples/
    └── multi_agent_training_example.py
```

### 3. **Update Imports**
Add to `janus/agents/__init__.py`:
```python
from .multi_agent_ppo import MultiAgentPPOAgent, MultiAgentNetworkConfig
```

Add to `janus/training/ppo/__init__.py`:
```python
from .multi_agent_buffer import MultiAgentRolloutBuffer
from .multi_agent_trainer import MultiAgentPPOTrainer
from .multi_agent_config import MultiAgentConfig
```

## Usage Examples

### Cooperative Scenario
```python
# Create configuration
config = MultiAgentConfig(
    n_agents=4,
    centralized_training=True,
    share_critic_params=True,
    reward_sharing="mixed",
    reward_sharing_weight=0.7,
    # ... other PPO parameters
)

# Create agents
agents = []
for i in range(config.n_agents):
    agent = MultiAgentPPOAgent(
        agent_id=i,
        n_agents=config.n_agents,
        observation_dim=obs_dim,
        action_dim=action_dim,
        actor_config=actor_config,
        critic_config=critic_config,
        device=config.device
    )
    agents.append(agent)

# Create trainer and train
trainer = MultiAgentPPOTrainer(
    agents=agents,
    envs=envs,
    config=config
)
trainer.train(total_timesteps=1000000)
```

### Competitive Scenario
```python
# Different configuration for competition
config = MultiAgentConfig(
    n_agents=2,
    centralized_training=True,
    share_critic_params=False,  # Separate critics
    reward_sharing="individual",
    # Lower learning rate for stability
    learning_rate=1e-4,
    clip_epsilon=0.1,
    target_kl=0.01
)
```

## Environment Requirements

Your multi-agent environments should follow this interface:

```python
class MultiAgentEnv:
    def reset(self) -> Tuple[List[np.ndarray], Dict]:
        """Returns list of observations for all agents."""
        pass

    def step(self, actions: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[bool], bool, Dict]:
        """
        Args:
            actions: Actions for all agents [n_agents, action_dim]
        Returns:
            observations: List of observations
            rewards: List of rewards
            dones: List of done flags
            truncated: Whether episode was truncated
            info: Additional information
        """
        pass
```

## Performance Considerations

### Memory Usage
- The centralized critic increases memory usage proportionally to the number of agents
- Use gradient accumulation for large agent counts
- Consider sharing critic parameters in cooperative scenarios

### Computational Efficiency
- Batch operations across agents when possible
- Use mixed precision training (`use_mixed_precision=True`)
- Parallelize environment stepping with `num_workers`

### Hyperparameter Tuning
- **Cooperative**: Higher learning rates, shared critics, mixed rewards
- **Competitive**: Lower learning rates, separate critics, individual rewards
- **Mixed**: Careful balance, consider agent groupings

## Advanced Features

### Communication (Future Extension)
The framework is prepared for inter-agent communication:
```python
config = MultiAgentConfig(
    enable_communication=True,
    communication_dim=32,
    communication_rounds=2
)
```

### Agent Grouping
For team-based scenarios:
```python
config = MultiAgentConfig(
    agent_groups={
        "team_1": [0, 1, 2],
        "team_2": [3, 4, 5]
    }
)
```

## Debugging Tips

1. **Start Simple**: Test with 2 agents before scaling up
2. **Monitor Individual Performance**: Use `log_agent_metrics=True`
3. **Check Advantage Estimates**: Ensure advantages are properly normalized
4. **Validate Critic Input**: Print shapes of joint observations/actions
5. **Compare Against Baselines**: Run independent PPO as a baseline

## Common Issues and Solutions

### Issue: Unstable Training
- Reduce learning rate
- Increase `target_kl` constraint
- Use separate optimizers for critics

### Issue: Agents Not Coordinating
- Increase `reward_sharing_weight` in mixed mode
- Enable communication (when implemented)
- Check if environment truly requires coordination

### Issue: Memory Errors
- Reduce `batch_size` or `rollout_length`
- Use gradient accumulation
- Limit number of parallel environments

## Next Steps

1. **Test with Simple Environments**: Start with the provided examples
2. **Implement Custom Environments**: Create domain-specific multi-agent tasks
3. **Experiment with Configurations**: Try different reward sharing schemes
4. **Add Communication**: Extend the framework with message passing
5. **Scale Up**: Test with larger agent populations

This multi-agent extension seamlessly integrates with your existing JanusAI_V2 framework while maintaining the modularity and production-quality standards of your codebase.
