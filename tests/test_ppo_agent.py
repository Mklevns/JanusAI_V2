# tests/test_ppo_agent.py
"""
Comprehensive unit tests for the production PPO Agent.
Run with: pytest tests/test_ppo_agent.py -v --cov=janus.agents.ppo_agent
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import yaml

from janus.agents.ppo_agent import (
    PPOAgent, NetworkConfig, MLP, RecurrentNetwork,
    init_weights, create_ppo_agent_from_yaml
)


class TestNetworkConfig:
    """Test NetworkConfig functionality."""
    
    def test_default_config(self):
        config = NetworkConfig(layer_sizes=[128, 64])
        assert config.activation == "tanh"
        assert not config.use_layer_norm
        assert config.initialization == "orthogonal"
        
    def test_invalid_activation(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            NetworkConfig(layer_sizes=[128], activation="invalid")
            
    def test_get_activation(self):
        activations = ["tanh", "relu", "gelu", "elu", "swish", "mish", "leaky_relu"]
        for act_name in activations:
            config = NetworkConfig(layer_sizes=[128], activation=act_name)
            activation = config.get_activation()
            assert isinstance(activation, nn.Module)
            
    def test_yaml_serialization(self):
        config = NetworkConfig(
            layer_sizes=[256, 128],
            activation="gelu",
            use_layer_norm=True,
            dropout_rate=0.1
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            loaded_config = NetworkConfig.from_yaml(f.name)
            
        assert loaded_config.layer_sizes == config.layer_sizes
        assert loaded_config.activation == config.activation
        assert loaded_config.use_layer_norm == config.use_layer_norm
        
    def test_recurrent_validation(self):
        # Valid recurrent config
        config = NetworkConfig(
            layer_sizes=[128],
            use_recurrent=True,
            recurrent_type="lstm"
        )
        assert config.recurrent_hidden_size == 128
        
        # Invalid recurrent type
        with pytest.raises(ValueError):
            NetworkConfig(
                layer_sizes=[128],
                use_recurrent=True,
                recurrent_type="invalid"
            )


class TestNetworkBuilding:
    """Test network construction utilities."""
    
    def test_init_weights(self):
        linear = nn.Linear(10, 20)
        
        # Test orthogonal initialization
        init_weights(linear, "orthogonal", gain=2.0)
        assert not torch.allclose(linear.weight, torch.zeros_like(linear.weight))
        
        # Test xavier initialization
        init_weights(linear, "xavier", gain=1.0)
        
        # Test kaiming initialization
        init_weights(linear, "kaiming")
        
    def test_mlp_construction(self):
        config = NetworkConfig(
            layer_sizes=[128, 64],
            activation="relu",
            use_layer_norm=True,
            dropout_rate=0.1
        )
        
        mlp = MLP(input_dim=10, output_dim=4, config=config)
        
        # Check structure
        assert isinstance(mlp.network[0], nn.Linear)  # First linear
        assert mlp.network[0].in_features == 10
        assert mlp.network[0].out_features == 128
        
        # Check for layer norm
        has_layer_norm = any(isinstance(m, nn.LayerNorm) for m in mlp.network)
        assert has_layer_norm
        
        # Check for dropout
        has_dropout = any(isinstance(m, nn.Dropout) for m in mlp.network)
        assert has_dropout
        
        # Test forward pass
        x = torch.randn(32, 10)
        output = mlp(x)
        assert output.shape == (32, 4)
        
    def test_recurrent_network(self):
        config = NetworkConfig(
            layer_sizes=[64],
            use_recurrent=True,
            recurrent_type="lstm",
            recurrent_hidden_size=128,
            recurrent_layers=2
        )
        
        rnn = RecurrentNetwork(input_dim=10, config=config)
        
        # Test single step
        x = torch.randn(1, 10)
        output, hidden = rnn(x)
        assert output.shape == (1, 1, 128)
        assert len(hidden) == 2  # (h, c) for LSTM
        
        # Test sequence
        x_seq = torch.randn(4, 20, 10)  # batch=4, seq_len=20
        output_seq, hidden_seq = rnn(x_seq, hidden)
        assert output_seq.shape == (4, 20, 128)


class TestPPOAgentCore:
    """Test core PPO Agent functionality."""
    
    @pytest.fixture
    def discrete_agent(self):
        """Create a discrete action agent."""
        actor_config = NetworkConfig(layer_sizes=[64, 64])
        critic_config = NetworkConfig(layer_sizes=[64, 64])
        
        agent = PPOAgent(
            observation_dim=8,
            action_dim=4,
            actor_config=actor_config,
            critic_config=critic_config,
            continuous_actions=False,
            seed=42
        )
        return agent
        
    @pytest.fixture
    def continuous_agent(self):
        """Create a continuous action agent."""
        actor_config = NetworkConfig(layer_sizes=[64, 64], activation="tanh")
        critic_config = NetworkConfig(layer_sizes=[64, 64], activation="relu")
        
        agent = PPOAgent(
            observation_dim=8,
            action_dim=2,
            actor_config=actor_config,
            critic_config=critic_config,
            continuous_actions=True,
            action_std_init=0.5,
            seed=42
        )
        return agent
        
    def test_initialization(self, discrete_agent):
        assert discrete_agent.observation_dim == 8
        assert discrete_agent.action_dim == 4
        assert not discrete_agent.continuous_actions
        assert discrete_agent.device.type in ['cpu', 'cuda']
        
    def test_seed_setting(self):
        PPOAgent.set_seed(123)
        
        # Test torch randomness
        t1 = torch.randn(5)
        PPOAgent.set_seed(123)
        t2 = torch.randn(5)
        assert torch.allclose(t1, t2)
        
    def test_discrete_action_selection(self, discrete_agent):
        obs = torch.randn(1, 8)
        action, log_prob, _ = discrete_agent.act(obs)
        
        assert action.shape == torch.Size([])  # Scalar
        assert log_prob.shape == torch.Size([])  # Scalar
        assert 0 <= action.item() < 4
        
    def test_discrete_batch_actions(self, discrete_agent):
        batch_size = 16
        obs = torch.randn(batch_size, 8)
        actions, log_probs, _ = discrete_agent.act(obs)
        
        assert actions.shape == (batch_size,)
        assert log_probs.shape == (batch_size,)
        assert all(0 <= a.item() < 4 for a in actions)
        
    def test_action_masking_single(self, discrete_agent):
        obs = torch.randn(1, 8)
        mask = torch.tensor([True, False, True, False])  # Only allow actions 0 and 2
        
        # Sample many times to verify masking
        sampled_actions = []
        for _ in range(100):
            action, _, _ = discrete_agent.act(obs, action_mask=mask)
            sampled_actions.append(action.item())
            
        # Check only valid actions were sampled
        unique_actions = set(sampled_actions)
        assert unique_actions.issubset({0, 2})
        assert len(unique_actions) == 2  # Both valid actions should be sampled
        
    def test_action_masking_batch(self, discrete_agent):
        batch_size = 8
        obs = torch.randn(batch_size, 8)
        
        # Different mask for each sample
        masks = torch.zeros(batch_size, 4, dtype=torch.bool)
        for i in range(batch_size):
            # Each sample has different valid actions
            valid_actions = np.random.choice(4, size=2, replace=False)
            masks[i, valid_actions] = True
            
        actions, _, _ = discrete_agent.act(obs, action_mask=masks)
        
        # Verify each action respects its mask
        for i in range(batch_size):
            assert masks[i, actions[i].item()].item() == True
            
    def test_continuous_actions(self, continuous_agent):
        obs = torch.randn(10, 8)
        actions, log_probs, _ = continuous_agent.act(obs)
        
        assert actions.shape == (10, 2)
        assert log_probs.shape == (10,)
        
        # Actions should be continuous values
        assert actions.dtype == torch.float32
        
    def test_deterministic_actions(self, discrete_agent):
        obs = torch.randn(1, 8)
        
        # Deterministic should always return same action
        action1, _, _ = discrete_agent.act(obs, deterministic=True)
        action2, _, _ = discrete_agent.act(obs, deterministic=True)
        assert action1.item() == action2.item()
        
    def test_evaluate_method(self, discrete_agent):
        obs = torch.randn(32, 8)
        actions = torch.randint(0, 4, (32,))
        
        log_probs, values, entropy = discrete_agent.evaluate(obs, actions)
        
        assert log_probs.shape == (32,)
        assert values.shape == (32,)
        assert entropy.shape == (32,)
        
        # Log probs should be negative
        assert (log_probs <= 0).all()
        
        # Entropy should be non-negative
        assert (entropy >= 0).all()
        
    def test_evaluate_with_masking(self, discrete_agent):
        obs = torch.randn(16, 8)
        actions = torch.randint(0, 4, (16,))
        masks = torch.ones(16, 4, dtype=torch.bool)
        masks[:, 3] = False  # Disable action 3 for all
        
        log_probs, values, entropy = discrete_agent.evaluate(obs, actions, action_mask=masks)
        
        # Actions that were masked out should have very low probability
        # (though they might still be evaluated if they were the actual actions taken)
        assert log_probs.shape == (16,)
        
    def test_get_value(self, discrete_agent):
        obs = torch.randn(1, 8)
        value, _ = discrete_agent.get_value(obs)
        
        assert value.shape == torch.Size([])  # Scalar for single obs
        assert value.dtype == torch.float32
        
        # Batch values
        obs_batch = torch.randn(10, 8)
        values, _ = discrete_agent.get_value(obs_batch)
        assert values.shape == (10,)
        
    def test_metrics_tracking(self, discrete_agent):
        # Reset metrics
        discrete_agent.reset_metrics()
        
        # Perform some actions
        obs = torch.randn(5, 8)
        discrete_agent.act(obs)
        discrete_agent.get_value(obs)
        
        metrics = discrete_agent.get_metrics()
        assert metrics['total_steps'] == 5
        assert len(metrics['action_entropy']) > 0
        assert len(metrics['value_predictions']) > 0
        assert 'avg_action_entropy' in metrics


class TestRecurrentAgent:
    """Test recurrent PPO agent functionality."""
    
    @pytest.fixture
    def recurrent_agent(self):
        actor_config = NetworkConfig(
            layer_sizes=[64],
            use_recurrent=True,
            recurrent_type="lstm",
            recurrent_hidden_size=128
        )
        critic_config = NetworkConfig(
            layer_sizes=[64, 32],
            use_recurrent=True,
            recurrent_type="gru",
            recurrent_hidden_size=64
        )
        
        agent = PPOAgent(
            observation_dim=10,
            action_dim=4,
            actor_config=actor_config,
            critic_config=critic_config,
            seed=42
        )
        return agent
        
    def test_hidden_state_initialization(self, recurrent_agent):
        hidden = recurrent_agent.reset_hidden_states(batch_size=4)
        
        # Actor uses LSTM (returns tuple)
        assert 'actor' in hidden
        assert isinstance(hidden['actor'], tuple)
        assert hidden['actor'][0].shape == (1, 4, 128)  # (layers, batch, hidden)
        
        # Critic uses GRU (returns single tensor)
        assert 'critic' in hidden
        assert isinstance(hidden['critic'], torch.Tensor)
        assert hidden['critic'].shape == (1, 4, 64)
        
    def test_recurrent_action_sequence(self, recurrent_agent):
        seq_len = 10
        hidden = recurrent_agent.reset_hidden_states(batch_size=1)
        
        actions = []
        for t in range(seq_len):
            obs = torch.randn(1, 10)
            action, log_prob, hidden['actor'] = recurrent_agent.act(
                obs, actor_hidden=hidden['actor']
            )
            actions.append(action.item())
            
        assert len(actions) == seq_len
        
    def test_recurrent_evaluation(self, recurrent_agent):
        batch_size = 8
        obs = torch.randn(batch_size, 10)
        actions = torch.randint(0, 4, (batch_size,))
        
        hidden = recurrent_agent.reset_hidden_states(batch_size)
        
        log_probs, values, entropy = recurrent_agent.evaluate(
            obs, actions,
            actor_hidden=hidden['actor'],
            critic_hidden=hidden['critic']
        )
        
        assert log_probs.shape == (batch_size,)
        assert values.shape == (batch_size,)


class TestCheckpointing:
    """Test model checkpointing functionality."""
    
    def test_save_load_discrete(self):
        # Create agent
        config = NetworkConfig(layer_sizes=[32, 32])
        agent = PPOAgent(
            observation_dim=4,
            action_dim=2,
            actor_config=config,
            critic_config=config,
            seed=42
        )
        
        # Get initial predictions
        obs = torch.randn(1, 4)
        action1, _, _ = agent.act(obs, deterministic=True)
        value1, _ = agent.get_value(obs)
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            agent.save_checkpoint(path, training_step=1000)
            
            # Create new agent and load
            new_agent = PPOAgent(
                observation_dim=4,
                action_dim=2,
                actor_config=config,
                critic_config=config
            )
            
            checkpoint = new_agent.load_checkpoint(path)
            
            # Verify checkpoint contents
            assert checkpoint['training_step'] == 1000
            assert checkpoint['observation_dim'] == 4
            
            # Verify predictions match
            action2, _, _ = new_agent.act(obs, deterministic=True)
            value2, _ = new_agent.get_value(obs)
            
            assert action1.item() == action2.item()
            assert torch.allclose(value1, value2)
            
    def test_save_load_continuous(self):
        config = NetworkConfig(layer_sizes=[32])
        agent = PPOAgent(
            observation_dim=4,
            action_dim=2,
            actor_config=config,
            critic_config=config,
            continuous_actions=True,
            action_std_init=0.3
        )
        
        # Modify log_std
        agent.log_std.data.fill_(-1.0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            agent.save_checkpoint(path)
            
            # Load into new agent
            new_agent = PPOAgent(
                observation_dim=4,
                action_dim=2,
                actor_config=config,
                critic_config=config,
                continuous_actions=True
            )
            new_agent.load_checkpoint(path)
            
            # Verify log_std was loaded
            assert torch.allclose(new_agent.log_std, agent.log_std)
            
    def test_incompatible_checkpoint(self):
        config = NetworkConfig(layer_sizes=[32])
        agent = PPOAgent(
            observation_dim=4,
            action_dim=2,
            actor_config=config,
            critic_config=config
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            agent.save_checkpoint(path)
            
            # Try to load into incompatible agent
            wrong_agent = PPOAgent(
                observation_dim=8,  # Different obs dim
                action_dim=2,
                actor_config=config,
                critic_config=config
            )
            
            with pytest.raises(ValueError, match="Observation dim mismatch"):
                wrong_agent.load_checkpoint(path)


class TestAdvancedFeatures:
    """Test advanced agent features."""
    
    def test_mixed_precision(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        config = NetworkConfig(layer_sizes=[64])
        agent = PPOAgent(
            observation_dim=8,
            action_dim=4,
            actor_config=config,
            critic_config=config,
            device='cuda',
            enable_amp=True
        )
        
        assert agent.enable_amp
        
        # Test that forward passes work with AMP
        obs = torch.randn(16, 8).cuda()
        with torch.cuda.amp.autocast():
            action, _, _ = agent.act(obs)
            value, _ = agent.get_value(obs)
            
    def test_spectral_norm(self):
        config = NetworkConfig(
            layer_sizes=[64, 64],
            use_spectral_norm=True
        )
        
        agent = PPOAgent(
            observation_dim=8,
            action_dim=4,
            actor_config=config,
            critic_config=config
        )
        
        # Check that spectral norm is applied
        for module in agent.actor.modules():
            if isinstance(module, nn.Linear):
                # Spectral norm adds a 'weight_orig' parameter
                has_spectral = hasattr(module, 'weight_orig')
                if has_spectral:
                    break
        else:
            pytest.fail("No spectral norm found in actor")
            
    def test_create_from_yaml(self):
        # Create config files
        actor_config = NetworkConfig(
            layer_sizes=[128, 64],
            activation="gelu",
            use_layer_norm=True
        )
        critic_config = NetworkConfig(
            layer_sizes=[128, 64],
            activation="tanh"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            actor_config.to_yaml(tmpdir / "actor_config.yaml")
            critic_config.to_yaml(tmpdir / "critic_config.yaml")
            
            # Create agent from YAML
            agent = create_ppo_agent_from_yaml(
                config_path=tmpdir,
                observation_dim=10,
                action_dim=4,
                continuous_actions=False
            )
            
            assert agent.observation_dim == 10
            assert agent.action_dim == 4
            assert agent.actor_config.activation == "gelu"
            assert agent.critic_config.activation == "tanh"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_valid_action(self):
        """Test when only one action is valid."""
        config = NetworkConfig(layer_sizes=[32])
        agent = PPOAgent(
            observation_dim=4,
            action_dim=4,
            actor_config=config,
            critic_config=config
        )
        
        obs = torch.randn(1, 4)
        mask = torch.tensor([False, False, True, False])  # Only action 2 valid
        
        for _ in range(10):
            action, _, _ = agent.act(obs, action_mask=mask)
            assert action.item() == 2
            
    def test_no_valid_actions(self):
        """Test behavior when no actions are valid (should not happen in practice)."""
        config = NetworkConfig(layer_sizes=[32])
        agent = PPOAgent(
            observation_dim=4,
            action_dim=4,
            actor_config=config,
            critic_config=config
        )
        
        obs = torch.randn(1, 4)
        mask = torch.tensor([False, False, False, False])  # No valid actions
        
        # Should still work but with undefined behavior
        # In practice, at least one action should always be valid
        action, log_prob, _ = agent.act(obs, action_mask=mask)
        assert not torch.isnan(log_prob)
        
    def test_device_consistency(self):
        """Test that all operations maintain device consistency."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        config = NetworkConfig(layer_sizes=[32])
        agent = PPOAgent(
            observation_dim=4,
            action_dim=2,
            actor_config=config,
            critic_config=config,
            device=device
        )
        
        # CPU observation should be moved to agent's device
        obs_cpu = torch.randn(4, 4)
        action, log_prob, _ = agent.act(obs_cpu)
        
        assert action.device.type == 'cpu'  # Outputs are moved back to CPU
        assert log_prob.device.type == 'cpu'
        
        # Direct device observation
        obs_device = torch.randn(4, 4).to(device)
        value, _ = agent.get_value(obs_device)
        assert value.device.type == device


if __name__ == '__main__':
    # Run specific test for debugging
    import sys
    
    if len(sys.argv) > 1:
        pytest.main([__file__, '-v', '-k', sys.argv[1]])
    else:
        pytest.main([__file__, '-v'])