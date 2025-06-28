# examples/ppo_agent_usage.py
"""
Comprehensive usage examples for the production PPO Agent.

Demonstrates various use cases including:
1. Basic discrete and continuous agents
2. Action masking for constrained environments
3. Recurrent agents for partial observability
4. Multi-agent scenarios
5. Custom environment integration
"""
import logging
import torch
import numpy as np
from pathlib import Path

from janus.agents.ppo_agent import PPOAgent, NetworkConfig, create_ppo_agent_from_yaml


# =============================================================================
# Example 1: Basic Discrete Action Agent
# =============================================================================

def example_discrete_agent():
    """Basic usage of discrete action PPO agent."""
    print("\n=== Example 1: Discrete Action Agent ===")
    
    # Create network configurations
    actor_config = NetworkConfig(
        layer_sizes=[128, 128],
        activation="tanh",
        initialization="orthogonal",
        gain=np.sqrt(2)
    )
    
    critic_config = NetworkConfig(
        layer_sizes=[128, 128],
        activation="relu",
        use_layer_norm=True
    )
    
    # Create agent
    agent = PPOAgent(
        observation_dim=8,
        action_dim=4,
        actor_config=actor_config,
        critic_config=critic_config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42
    )
    
    # Single observation
    obs = torch.randn(8)
    action, log_prob, _ = agent.act(obs)
    print(f"Single action: {action.item()}, Log prob: {log_prob.item():.3f}")
    
    # Batch observations
    batch_obs = torch.randn(16, 8)
    actions, log_probs, _ = agent.act(batch_obs)
    print(f"Batch actions shape: {actions.shape}")
    
    # Get value estimates
    values, _ = agent.get_value(batch_obs)
    print(f"Value estimates: min={values.min():.3f}, max={values.max():.3f}")
    
    return agent


# =============================================================================
# Example 2: Continuous Control Agent
# =============================================================================

def example_continuous_agent():
    """Continuous action space agent for control tasks."""
    print("\n=== Example 2: Continuous Control Agent ===")
    
    # Create agent for continuous control
    agent = PPOAgent(
        observation_dim=24,  # e.g., joint positions, velocities
        action_dim=6,        # e.g., joint torques
        actor_config=NetworkConfig(
            layer_sizes=[256, 256],
            activation="tanh",
            use_layer_norm=True
        ),
        critic_config=NetworkConfig(
            layer_sizes=[256, 256],
            activation="relu"
        ),
        continuous_actions=True,
        action_std_init=0.3,  # Initial exploration noise
        enable_amp=torch.cuda.is_available()  # Use mixed precision if available
    )
    
    # Sample continuous actions
    obs = torch.randn(10, 24)
    actions, log_probs, _ = agent.act(obs)
    
    print(f"Continuous actions shape: {actions.shape}")
    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"Action std: {actions.std():.3f}")
    
    # Deterministic policy (for evaluation)
    det_actions, _, _ = agent.act(obs, deterministic=True)
    print(f"Deterministic action std: {det_actions.std():.3f} (should be 0)")
    
    return agent


# =============================================================================
# Example 3: Action Masking for Constrained Environments
# =============================================================================

def example_action_masking():
    """Demonstrate action masking for environments with constraints."""
    print("\n=== Example 3: Action Masking ===")
    
    # Create agent with 6 possible actions
    agent = PPOAgent(
        observation_dim=10,
        action_dim=6,
        actor_config=NetworkConfig(layer_sizes=[64, 64]),
        critic_config=NetworkConfig(layer_sizes=[64, 64]),
        seed=42
    )
    
    # Scenario 1: Fixed mask (e.g., some actions always invalid)
    obs = torch.randn(1, 10)
    mask = torch.tensor([True, True, False, True, False, True])  # Actions 2,4 invalid
    
    print("Fixed mask scenario:")
    for i in range(5):
        action, _, _ = agent.act(obs, action_mask=mask)
        print(f"  Action {i+1}: {action.item()} (valid: {mask[action].item()})")
    
    # Scenario 2: Dynamic masks per observation
    batch_size = 8
    batch_obs = torch.randn(batch_size, 10)
    
    # Different constraints for each observation
    batch_masks = torch.zeros(batch_size, 6, dtype=torch.bool)
    for i in range(batch_size):
        # Randomly disable 2-3 actions per observation
        num_valid = np.random.randint(3, 5)
        valid_indices = np.random.choice(6, num_valid, replace=False)
        batch_masks[i, valid_indices] = True
    
    print("\nDynamic mask scenario:")
    actions, _, _ = agent.act(batch_obs, action_mask=batch_masks)
    
    for i in range(batch_size):
        valid_actions = torch.where(batch_masks[i])[0].tolist()
        print(f"  Obs {i}: valid actions={valid_actions}, selected={actions[i].item()}")
        assert actions[i].item() in valid_actions
    
    return agent


# =============================================================================
# Example 4: Recurrent Agent for Partial Observability
# =============================================================================

def example_recurrent_agent():
    """Recurrent agent for environments with partial observability."""
    print("\n=== Example 4: Recurrent Agent ===")
    
    # Create LSTM-based agent
    agent = PPOAgent(
        observation_dim=10,
        action_dim=4,
        actor_config=NetworkConfig(
            layer_sizes=[64],
            use_recurrent=True,
            recurrent_type="lstm",
            recurrent_hidden_size=128,
            recurrent_layers=2
        ),
        critic_config=NetworkConfig(
            layer_sizes=[128, 64],
            use_recurrent=True,
            recurrent_type="gru",  # Can use different architectures
            recurrent_hidden_size=64
        )
    )
    
    # Process a sequence of observations
    sequence_length = 20
    batch_size = 4
    
    # Initialize hidden states for batch
    hidden_states = agent.reset_hidden_states(batch_size)
    
    print(f"Processing sequence of length {sequence_length}:")
    
    total_rewards = np.zeros(batch_size)
    for t in range(sequence_length):
        # Get observations for this timestep
        obs = torch.randn(batch_size, 10)
        
        # Act using current hidden state
        actions, log_probs, hidden_states['actor'] = agent.act(
            obs, 
            actor_hidden=hidden_states['actor']
        )
        
        # Get values using hidden state
        values, hidden_states['critic'] = agent.get_value(
            obs,
            critic_hidden=hidden_states['critic']
        )
        
        # Simulate rewards (higher for consistent actions)
        if t > 0:
            rewards = (actions == prev_actions).float() * 0.1 + np.random.randn(batch_size) * 0.05
            total_rewards += rewards.numpy()
        
        prev_actions = actions
        
        if t % 5 == 0:
            print(f"  Step {t}: Actions={actions.tolist()}, "
                  f"Avg value={values.mean():.3f}")
    
    print(f"Total rewards: {total_rewards}")
    
    return agent


# =============================================================================
# Example 5: Multi-Agent Setup
# =============================================================================

def example_multi_agent():
    """Multiple agents with shared or separate networks."""
    print("\n=== Example 5: Multi-Agent Setup ===")
    
    num_agents = 3
    shared_config = NetworkConfig(
        layer_sizes=[128, 128],
        activation="relu",
        use_layer_norm=True
    )
    
    # Create agents with shared architecture but separate parameters
    agents = []
    for i in range(num_agents):
        agent = PPOAgent(
            observation_dim=12,
            action_dim=4,
            actor_config=shared_config,
            critic_config=shared_config,
            seed=42 + i  # Different seeds for diversity
        )
        agents.append(agent)
    
    # Simulate multi-agent interaction
    print("Multi-agent interaction:")
    
    # Each agent observes the environment + other agents
    base_obs = torch.randn(12)
    
    for step in range(5):
        all_actions = []
        all_values = []
        
        for i, agent in enumerate(agents):
            # Could include other agents' previous actions in observation
            obs = base_obs  # Simplified
            
            action, _, _ = agent.act(obs, deterministic=True)
            value, _ = agent.get_value(obs)
            
            all_actions.append(action.item())
            all_values.append(value.item())
        
        print(f"  Step {step}: Actions={all_actions}, "
              f"Values={[f'{v:.3f}' for v in all_values]}")
        
        # Update base observation (simplified)
        base_obs = torch.randn(12)
    
    return agents


# =============================================================================
# Example 6: Integration with Custom Environment
# =============================================================================

class CustomEnv:
    """Example custom environment with PPO agent."""
    
    def __init__(self):
        self.observation_dim = 6
        self.action_dim = 3
        self.current_state = np.zeros(self.observation_dim)
        self.step_count = 0
        
    def reset(self):
        self.current_state = np.random.randn(self.observation_dim) * 0.1
        self.step_count = 0
        return self.current_state
    
    def step(self, action):
        # Simple dynamics
        self.current_state += np.random.randn(self.observation_dim) * 0.1
        self.current_state[action] += 0.5  # Action effect
        
        # Reward encourages selecting different actions
        reward = -0.1 * np.abs(self.current_state).sum()
        if self.step_count > 0 and action != self.last_action:
            reward += 1.0
            
        self.last_action = action
        self.step_count += 1
        done = self.step_count >= 50
        
        return self.current_state, reward, done
    
    def get_valid_actions(self):
        """Return mask of currently valid actions."""
        # Example: action 2 becomes invalid after 25 steps
        if self.step_count < 25:
            return torch.tensor([True, True, True])
        else:
            return torch.tensor([True, True, False])


def example_custom_environment():
    """Integrate PPO agent with custom environment."""
    print("\n=== Example 6: Custom Environment Integration ===")
    
    env = CustomEnv()
    
    # Create agent matching environment specs
    agent = PPOAgent(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        actor_config=NetworkConfig(
            layer_sizes=[64, 32],
            activation="tanh"
        ),
        critic_config=NetworkConfig(
            layer_sizes=[64, 32],
            activation="tanh"
        )
    )
    
    # Run episode
    print("Running episode with custom environment:")
    
    obs = env.reset()
    total_reward = 0
    
    for step in range(100):
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(obs)
        
        # Get valid actions mask
        action_mask = env.get_valid_actions()
        
        # Select action
        action, log_prob, _ = agent.act(obs_tensor, action_mask=action_mask)
        
        # Step environment
        obs, reward, done = env.step(action.item())
        total_reward += reward
        
        if step % 20 == 0:
            value, _ = agent.get_value(obs_tensor)
            print(f"  Step {step}: Action={action.item()}, "
                  f"Reward={reward:.3f}, Value={value.item():.3f}")
        
        if done:
            break
    
    print(f"Episode finished. Total reward: {total_reward:.2f}")
    
    return agent, env


# =============================================================================
# Example 7: Advanced Configuration and Checkpointing
# =============================================================================

def example_advanced_usage():
    """Advanced features including checkpointing and config management."""
    print("\n=== Example 7: Advanced Usage ===")
    
    # Create and save configurations
    import tempfile
    with tempfile.TemporaryDirectory() as config_dir:
        config_dir = Path(config_dir)
        
        # Advanced actor configuration
        actor_config = NetworkConfig(
            layer_sizes=[256, 256, 128],
            activation="gelu",
            use_layer_norm=True,
            use_spectral_norm=True,
            dropout_rate=0.1,
            initialization="xavier"
        )
        actor_config.to_yaml(config_dir / "actor_config.yaml")
        
        # Advanced critic configuration
        critic_config = NetworkConfig(
            layer_sizes=[256, 128],
            activation="elu",
            use_batch_norm=True,
            initialization="kaiming"
        )
        critic_config.to_yaml(config_dir / "critic_config.yaml")
        
        # Create agent from YAML configs
        agent = create_ppo_agent_from_yaml(
            config_path=config_dir,
            observation_dim=20,
            action_dim=8,
            continuous_actions=False,
            enable_amp=True
        )
        
        print(f"Created agent from YAML configs")
        print(f"  Actor: {agent.actor}")
        print(f"  Critic: {agent.critic}")
        
        # Train for a bit (simulated)
        for i in range(100):
            obs = torch.randn(32, 20)
            actions, _, _ = agent.act(obs)
            # Simulate training...
            
        # Save checkpoint
        checkpoint_path = config_dir / "model_checkpoint.pt"
        agent.save_checkpoint(
            checkpoint_path,
            epoch=10,
            global_step=1000,
            best_reward=95.5,
            optimizer_state={"lr": 1e-4}  # Can save additional info
        )
        
        print(f"\nSaved checkpoint with metrics")
        
        # Load checkpoint
        new_agent = create_ppo_agent_from_yaml(
            config_path=config_dir,
            observation_dim=20,
            action_dim=8
        )
        
        checkpoint_data = new_agent.load_checkpoint(checkpoint_path)
        print(f"\nLoaded checkpoint:")
        print(f"  Epoch: {checkpoint_data.get('epoch')}")
        print(f"  Global step: {checkpoint_data.get('global_step')}")
        print(f"  Best reward: {checkpoint_data.get('best_reward')}")
        
    return agent


# =============================================================================
# Main Runner
# =============================================================================

def run_all_examples():
    """Run all example demonstrations."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("PPO Agent Usage Examples")
    print("=" * 80)
    
    # Run examples
    discrete_agent = example_discrete_agent()
    continuous_agent = example_continuous_agent()
    masked_agent = example_action_masking()
    recurrent_agent = example_recurrent_agent()
    multi_agents = example_multi_agent()
    custom_agent, custom_env = example_custom_environment()
    advanced_agent = example_advanced_usage()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
    
    # Display final metrics
    print("\nFinal agent metrics:")
    metrics = discrete_agent.get_metrics()
    for key, value in metrics.items():
        if not isinstance(value, list):
            print(f"  {key}: {value}")


if __name__ == '__main__':
    run_all_examples()