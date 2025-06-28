# tests/test_ppo_integration.py
"""
Integration tests for the modular PPO implementation.

Tests that all components work together correctly and can run
a short training session without errors.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import torch
import numpy as np
import logging

from janus.training.ppo.config import PPOConfig
from janus.training.ppo.trainer import PPOTrainer
from janus.training.ppo.buffer import RolloutBuffer
from janus.training.ppo.collector import AsyncRolloutCollector
from janus.training.ppo.normalization import RunningMeanStd
from janus.agents.ppo_agent import PPOAgent, NetworkConfig
from janus.envs.symbolic_regression import SymbolicRegressionEnv

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestPPOComponents:
    """Test individual PPO components."""
    
    def test_config_loading(self, tmp_path):
        """Test configuration loading and saving."""
        config = PPOConfig(
            learning_rate=1e-3,
            n_epochs=5,
            batch_size=32,
            normalize_rewards=True
        )
        
        # Save to YAML
        config_path = tmp_path / "test_config.yaml"
        config.to_yaml(config_path)
        
        # Load from YAML
        loaded_config = PPOConfig.from_yaml(config_path)
        
        assert loaded_config.learning_rate == 1e-3
        assert loaded_config.n_epochs == 5
        assert loaded_config.normalize_rewards
        
    def test_buffer_operations(self):
        """Test rollout buffer functionality."""
        buffer = RolloutBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_shape=(),
            device=torch.device('cpu'),
            n_envs=2
        )
        
        # Add data
        for i in range(50):
            buffer.add(
                obs=np.random.randn(2, 4),
                action=np.array([0, 1]),
                reward=np.array([1.0, -0.5]),
                done=np.array([False, False]),
                value=np.array([0.5, 0.3]),
                log_prob=np.array([-0.5, -0.7])
            )
            
        assert buffer.ptr == 50
        
        # Get data
        data = buffer.get()
        assert data['observations'].shape == (50, 2, 4)
        assert data['rewards'].shape == (50, 2)
        
        # Clear buffer
        buffer.clear()
        assert buffer.ptr == 0
        
    def test_normalization(self):
        """Test running mean/std normalization."""
        normalizer = RunningMeanStd(shape=(2,))
        
        # Update with batches
        for _ in range(10):
            data = np.random.randn(100, 2)
            normalizer.update(data)
            
        # Check statistics are reasonable
        assert normalizer.count > 1
        assert normalizer.mean.shape == (2,)
        assert normalizer.var.shape == (2,)
        assert np.all(normalizer.var > 0)
        
    def test_collector_initialization(self):
        """Test async rollout collector."""
        envs = [SymbolicRegressionEnv() for _ in range(2)]
        collector = AsyncRolloutCollector(
            envs, 
            torch.device('cpu'),
            num_workers=2
        )
        
        assert collector.n_envs == 2
        assert collector.current_obs.shape[0] == 2
        
        # Test step collection
        def get_actions_fn(obs_tensor):
            num_envs = obs_tensor.shape[0]
            # SymbolicRegressionEnv has a Discrete action space
            actions = np.array([envs[i].action_space.sample() for i in range(num_envs)])
            log_probs = np.zeros(num_envs) # Dummy log_probs
            return actions, log_probs

        # Call collect instead of collect_steps
        # New return signature: original_obs, next_obs_tensor, rewards_tensor, dones_tensor, truncateds_tensor, log_probs_np, infos_list
        original_obs, next_obs, rewards, dones, truncateds, log_probs, infos = collector.collect(get_actions_fn)
        
        assert original_obs.shape[0] == 2
        assert next_obs.shape[0] == 2 # torch.Tensor
        assert rewards.shape == (2,) # torch.Tensor
        assert dones.shape == (2,) # torch.Tensor
        assert truncateds.shape == (2,) # torch.Tensor
        assert log_probs.shape == (2,)
        assert isinstance(infos, list)
        assert len(infos) == 2
        assert isinstance(infos[0], dict)


class TestPPOIntegration:
    """Test full PPO training integration."""
    
    @pytest.fixture
    def setup_training(self):
        """Setup training components."""
        # Create simple config
        config = PPOConfig(
            learning_rate=3e-4,
            n_epochs=2,
            batch_size=32,
            num_workers=2,
            log_interval=1,
            save_interval=5,
            eval_interval=5,
            normalize_rewards=True,
            use_mixed_precision=False
        )
        
        # Create environments
        envs = [SymbolicRegressionEnv() for _ in range(2)]
        eval_env = SymbolicRegressionEnv()
        
        # Create agent
        obs_dim = envs[0].observation_space.shape[0]
        action_dim = envs[0].action_space.n
        
        actor_config = NetworkConfig(layer_sizes=[64, 64])
        critic_config = NetworkConfig(layer_sizes=[64, 64])
        
        agent = PPOAgent(
            observation_dim=obs_dim,
            action_dim=action_dim,
            actor_config=actor_config,
            critic_config=critic_config,
            seed=42
        )
        
        return config, envs, eval_env, agent
        
    def test_trainer_initialization(self, setup_training):
        """Test trainer can be initialized properly."""
        config, envs, eval_env, agent = setup_training
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(
                agent=agent,
                envs=envs,
                config=config,
                experiment_name="test_exp",
                checkpoint_dir=Path(tmpdir),
                use_tensorboard=False,
                use_wandb=False,
                eval_env=eval_env
            )
            
            assert trainer.n_envs == 2
            assert trainer.global_step == 0
            assert trainer.device.type in ['cpu', 'cuda']
            
    def test_short_training(self, setup_training):
        """Test a short training run completes without errors."""
        config, envs, eval_env, agent = setup_training
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(
                agent=agent,
                envs=envs,
                config=config,
                experiment_name="test_training",
                checkpoint_dir=Path(tmpdir),
                use_tensorboard=False,
                use_wandb=False,
                eval_env=eval_env
            )
            
            # Run short training
            trainer.train(
                total_timesteps=1000,
                rollout_length=100
            )
            
            # Check training progressed
            assert trainer.global_step >= 1000
            assert trainer.num_updates > 0
            
            # Check checkpoint was saved
            final_checkpoint = Path(tmpdir) / "final_model.pt"
            assert final_checkpoint.exists()
            
    def test_rollout_collection(self, setup_training):
        """Test rollout collection works correctly."""
        config, envs, eval_env, agent = setup_training
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(
                agent=agent,
                envs=envs,
                config=config,
                experiment_name="test_rollout",
                checkpoint_dir=Path(tmpdir),
                use_tensorboard=False,
                use_wandb=False
            )
            
            # Initialize buffer
            trainer.buffer = RolloutBuffer(
                buffer_size=100,
                obs_shape=trainer.obs_shape,
                action_shape=trainer.action_shape,
                device=trainer.device,
                n_envs=trainer.n_envs
            )
            
            # Collect rollouts
            metrics = trainer.collect_rollouts(100)
            
            assert 'mean_episode_reward' in metrics
            assert 'steps_per_second' in metrics
            assert trainer.buffer.ptr == 100
            
    def test_learning_step(self, setup_training):
        """Test learning from collected data."""
        config, envs, eval_env, agent = setup_training
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(
                agent=agent,
                envs=envs,
                config=config,
                experiment_name="test_learning",
                checkpoint_dir=Path(tmpdir),
                use_tensorboard=False,
                use_wandb=False
            )
            
            # Initialize buffer and collect data
            trainer.buffer = RolloutBuffer(
                buffer_size=200,
                obs_shape=trainer.obs_shape,
                action_shape=trainer.action_shape,
                device=trainer.device,
                n_envs=trainer.n_envs
            )
            
            trainer.collect_rollouts(200)
            
            # Learn from data
            metrics = trainer.learn(
                current_clip_epsilon=0.2,
                current_entropy_coef=0.01
            )
            
            assert 'policy_loss' in metrics
            assert 'value_loss' in metrics
            assert 'entropy' in metrics
            assert 'kl_divergence' in metrics
            assert metrics['policy_loss'] is not None
            
    def test_checkpoint_save_load(self, setup_training):
        """Test checkpoint saving and loading."""
        config, envs, eval_env, agent = setup_training
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create trainer
            trainer = PPOTrainer(
                agent=agent,
                envs=envs,
                config=config,
                experiment_name="test_checkpoint",
                checkpoint_dir=tmpdir,
                use_tensorboard=False,
                use_wandb=False
            )
            
            # Modify some state
            trainer.global_step = 1000
            trainer.best_mean_reward = 95.5
            
            # Save checkpoint
            checkpoint_path = trainer.save_checkpoint()
            assert checkpoint_path.exists()
            
            # Create new trainer and load
            new_trainer = PPOTrainer(
                agent=agent,
                envs=envs,
                config=config,
                experiment_name="test_checkpoint_2",
                checkpoint_dir=tmpdir,
                use_tensorboard=False,
                use_wandb=False
            )
            
            # Load checkpoint
            new_trainer.load_checkpoint(checkpoint_path)
            
            assert new_trainer.global_step == 1000
            assert new_trainer.best_mean_reward == 95.5
            
    def test_evaluation(self, setup_training):
        """Test model evaluation."""
        config, envs, eval_env, agent = setup_training
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(
                agent=agent,
                envs=envs,
                config=config,
                experiment_name="test_eval",
                checkpoint_dir=Path(tmpdir),
                use_tensorboard=False,
                use_wandb=False,
                eval_env=eval_env
            )
            
            # Run evaluation
            eval_metrics = trainer.evaluate(num_episodes=5)
            
            assert 'eval_reward' in eval_metrics
            assert 'eval_reward_std' in eval_metrics
            assert 'eval_length' in eval_metrics
            assert isinstance(eval_metrics['eval_reward'], float)


class TestErrorHandling:
    """Test error handling and recovery."""
    
    def test_training_interruption(self, setup_training):
        """Test that training handles interruption gracefully."""
        config, envs, eval_env, agent = setup_training
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PPOTrainer(
                agent=agent,
                envs=envs,
                config=config,
                experiment_name="test_interrupt",
                checkpoint_dir=Path(tmpdir),
                use_tensorboard=False,
                use_wandb=False
            )
            
            # Simulate keyboard interrupt after some steps
            original_collect = trainer.collect_rollouts
            call_count = 0
            
            def mock_collect(num_steps):
                nonlocal call_count
                call_count += 1
                if call_count > 2:
                    raise KeyboardInterrupt()
                return original_collect(num_steps)
                
            trainer.collect_rollouts = mock_collect
            
            # Should handle interruption gracefully
            trainer.train(total_timesteps=10000, rollout_length=100)
            
            # Should have saved final model
            final_checkpoint = Path(tmpdir) / "final_model.pt"
            assert final_checkpoint.exists()


def run_quick_test():
    """Run a quick integration test for debugging."""
    print("Running quick PPO integration test...")
    
    # Setup
    config = PPOConfig(
        learning_rate=3e-4,
        n_epochs=2,
        batch_size=32,
        num_workers=2
    )
    
    envs = [SymbolicRegressionEnv() for _ in range(2)]
    obs_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.n
    
    agent = PPOAgent(
        observation_dim=obs_dim,
        action_dim=action_dim,
        actor_config=NetworkConfig(layer_sizes=[64, 64]),
        critic_config=NetworkConfig(layer_sizes=[64, 64])
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = PPOTrainer(
            agent=agent,
            envs=envs,
            config=config,
            experiment_name="quick_test",
            checkpoint_dir=Path(tmpdir),
            use_tensorboard=False,
            use_wandb=False
        )
        
        # Train for a few steps
        trainer.train(total_timesteps=400, rollout_length=100)
        
    print("✓ Quick test completed successfully!")


if __name__ == '__main__':
    run_quick_test()