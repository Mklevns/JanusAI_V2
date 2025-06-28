# tests/test_async_ppo.py
"""
Unit tests for async PPO functionality.
Run with: pytest tests/test_async_ppo.py -v
"""
import pytest
import numpy as np
import torch
import gym
import threading
import time
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor

from janus.training.ppo_trainer import (
    PPOConfig, PPOTrainer, AsyncRolloutCollector, 
    RolloutBuffer, RunningMeanStd
)
from janus.agents.ppo_agent import PPOAgent


class TestAsyncRolloutCollector:
    """Test the async rollout collector."""
    
    @pytest.fixture
    def setup_collector(self):
        """Create a basic collector setup."""
        envs = [gym.make('CartPole-v1') for _ in range(4)]
        device = torch.device('cpu')
        collector = AsyncRolloutCollector(envs, device, num_workers=2)
        return collector, envs
        
    def test_initialization(self, setup_collector):
        collector, envs = setup_collector
        
        assert collector.n_envs == 4
        assert collector.num_workers == 2
        assert collector.current_obs.shape == (4, 4)  # 4 envs, 4 obs dims
        assert len(collector.episode_rewards) == 4
        
    def test_reset_all_envs(self, setup_collector):
        collector, envs = setup_collector
        
        obs = collector._reset_all_envs()
        assert obs.shape == (4, 4)
        assert np.all(collector.last_valid_obs[0] == obs[0])
        
    def test_step_env_safe_success(self, setup_collector):
        collector, envs = setup_collector
        
        action = envs[0].action_space.sample()
        next_obs, reward, done, truncated = collector._step_env_safe(0, action)
        
        assert next_obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        
    def test_step_env_safe_with_error(self, setup_collector):
        collector, envs = setup_collector
        
        # Mock environment to fail
        with patch.object(envs[0], 'step', side_effect=Exception("Test error")):
            next_obs, reward, done, truncated = collector._step_env_safe(0, 0, max_retries=2)
            
        # Should return last valid obs and done=True
        assert np.array_equal(next_obs, collector.last_valid_obs[0])
        assert reward == 0.0
        assert done == True
        
    def test_collect_steps_sync(self, setup_collector):
        collector, envs = setup_collector
        
        actions = np.array([env.action_space.sample() for env in envs])
        next_obs, rewards, dones, truncateds = collector.collect_steps(actions)
        
        assert next_obs.shape == (4, 4)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        
    def test_collect_steps_async(self, setup_collector):
        collector, envs = setup_collector
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            actions = np.array([env.action_space.sample() for env in envs])
            next_obs, rewards, dones, truncateds = collector.collect_steps(actions, executor)
            
        assert next_obs.shape == (4, 4)
        assert rewards.shape == (4,)
        
    def test_episode_statistics(self, setup_collector):
        collector, envs = setup_collector
        
        # Run some episodes
        for _ in range(50):
            actions = np.array([env.action_space.sample() for env in envs])
            collector.collect_steps(actions)
            
        stats = collector.get_statistics()
        
        assert 'mean_episode_reward' in stats
        assert 'mean_episode_length' in stats
        assert stats['mean_episode_reward'] > 0  # CartPole should have positive rewards
        
    def test_thread_safety(self, setup_collector):
        collector, envs = setup_collector
        
        # Test concurrent access to statistics
        results = []
        
        def update_stats(env_idx):
            with collector.lock:
                collector.current_episode_rewards[env_idx] += 1.0
                collector.episode_rewards[env_idx].append(
                    collector.current_episode_rewards[env_idx]
                )
                
        threads = []
        for i in range(4):
            for j in range(10):
                t = threading.Thread(target=update_stats, args=(i,))
                threads.append(t)
                t.start()
                
        for t in threads:
            t.join()
            
        # Check all updates were recorded
        for i in range(4):
            assert len(collector.episode_rewards[i]) == 10


class TestThreadSafeComponents:
    """Test thread safety of shared components."""
    
    def test_running_mean_std_thread_safety(self):
        rms = RunningMeanStd()
        
        def update_rms(data):
            rms.update(data)
            
        # Concurrent updates
        threads = []
        for i in range(10):
            data = np.random.randn(100, 1)
            t = threading.Thread(target=update_rms, args=(data,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # Check count is correct
        assert rms.count == 1000 + 1e-4  # 10 threads * 100 samples + epsilon
        
    def test_rollout_buffer_thread_safety(self):
        buffer = RolloutBuffer(
            buffer_size=100,
            obs_shape=(4,),
            action_shape=(),
            device=torch.device('cpu'),
            n_envs=4
        )
        
        def add_to_buffer(step_idx):
            obs = np.ones((4, 4)) * step_idx
            action = np.ones(4) * step_idx
            buffer.add(obs, action, np.ones(4), np.zeros(4), np.ones(4), np.ones(4))
            
        # Concurrent adds
        threads = []
        for i in range(50):
            t = threading.Thread(target=add_to_buffer, args=(i,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # Check all adds succeeded (up to buffer size)
        assert buffer.ptr == 50


class TestAsyncPPOTrainer:
    """Test the full async PPO trainer."""
    
    @pytest.fixture
    def setup_trainer(self):
        """Create a basic trainer setup."""
        envs = [gym.make('CartPole-v1') for _ in range(2)]
        agent = PPOAgent(observation_dim=4, action_dim=2)
        config = PPOConfig(
            n_epochs=2, 
            batch_size=32,
            num_workers=2,
            use_subprocess_envs=False
        )
        trainer = PPOTrainer(
            agent, envs, config, 
            use_tensorboard=False, 
            use_wandb=False
        )
        return trainer
        
    def test_async_trainer_initialization(self, setup_trainer):
        trainer = setup_trainer
        
        assert trainer.n_envs == 2
        assert trainer.collector is not None
        assert trainer.executor is not None
        assert trainer.executor._max_workers == 2
        
    def test_async_rollout_collection(self, setup_trainer):
        trainer = setup_trainer
        
        # Initialize buffer
        trainer.buffer = RolloutBuffer(
            buffer_size=64,
            obs_shape=(4,),
            action_shape=(),
            device=trainer.device,
            n_envs=2
        )
        
        # Collect rollouts
        start_time = time.time()
        metrics = trainer.collect_rollouts(64)
        collection_time = time.time() - start_time
        
        assert 'steps_per_second' in metrics
        assert metrics['steps_per_second'] > 0
        assert trainer.buffer.ptr == 64
        
        print(f"Async collection: {metrics['steps_per_second']:.1f} steps/sec")
        
    def test_yaml_config_integration(self, tmp_path):
        """Test YAML config save/load with async settings."""
        config = PPOConfig(
            learning_rate=5e-4,
            num_workers=4,
            use_subprocess_envs=True,
            experiment_tags={'test': 'async'}
        )
        
        # Save to YAML
        yaml_path = tmp_path / "test_config.yaml"
        config.to_yaml(yaml_path)
        
        # Load from YAML
        loaded_config = PPOConfig.from_yaml(yaml_path)
        
        assert loaded_config.num_workers == 4
        assert loaded_config.use_subprocess_envs == True
        assert loaded_config.experiment_tags['test'] == 'async'
        
    def test_performance_vs_sequential(self):
        """Compare async vs sequential performance."""
        n_envs = 4
        rollout_length = 128
        
        # Sequential setup
        seq_config = PPOConfig(num_workers=1)
        seq_envs = [gym.make('CartPole-v1') for _ in range(n_envs)]
        seq_agent = PPOAgent(observation_dim=4, action_dim=2)
        seq_trainer = PPOTrainer(
            seq_agent, seq_envs, seq_config,
            use_tensorboard=False
        )
        seq_trainer.buffer = RolloutBuffer(
            rollout_length, (4,), (), torch.device('cpu'), n_envs
        )
        
        # Async setup
        async_config = PPOConfig(num_workers=4)
        async_envs = [gym.make('CartPole-v1') for _ in range(n_envs)]
        async_agent = PPOAgent(observation_dim=4, action_dim=2)
        async_trainer = PPOTrainer(
            async_agent, async_envs, async_config,
            use_tensorboard=False
        )
        async_trainer.buffer = RolloutBuffer(
            rollout_length, (4,), (), torch.device('cpu'), n_envs
        )
        
        # Benchmark sequential
        start = time.time()
        seq_metrics = seq_trainer.collect_rollouts(rollout_length)
        seq_time = time.time() - start
        
        # Benchmark async
        start = time.time()
        async_metrics = async_trainer.collect_rollouts(rollout_length)
        async_time = time.time() - start
        
        # Async should be faster (at least not slower)
        speedup = seq_time / async_time
        print(f"\nPerformance comparison:")
        print(f"Sequential: {seq_metrics['steps_per_second']:.1f} steps/sec")
        print(f"Async: {async_metrics['steps_per_second']:.1f} steps/sec")
        print(f"Speedup: {speedup:.2f}x")
        
        assert async_metrics['steps_per_second'] >= seq_metrics['steps_per_second'] * 0.9  # Allow 10% variance


class TestErrorHandling:
    """Test error handling in async operations."""
    
    def test_env_failure_recovery(self):
        """Test recovery from environment failures."""
        
        class FailingEnv(gym.Env):
            """Environment that fails after N steps."""
            def __init__(self, fail_after=10):
                self.observation_space = gym.spaces.Box(-1, 1, shape=(4,))
                self.action_space = gym.spaces.Discrete(2)
                self.fail_after = fail_after
                self.step_count = 0
                
            def reset(self):
                self.step_count = 0
                return np.zeros(4), {}
                
            def step(self, action):
                self.step_count += 1
                if self.step_count > self.fail_after:
                    raise RuntimeError("Environment failed!")
                return np.zeros(4), 1.0, False, False, {}
                
        # Create trainer with failing environments
        envs = [FailingEnv(fail_after=20) for _ in range(4)]
        agent = PPOAgent(observation_dim=4, action_dim=2)
        config = PPOConfig(
            num_workers=2,
            fail_on_env_error=False,  # Should recover
            max_env_retries=3
        )
        
        trainer = PPOTrainer(
            agent, envs, config,
            use_tensorboard=False
        )
        
        trainer.buffer = RolloutBuffer(50, (4,), (), torch.device('cpu'), 4)
        
        # Should complete despite failures
        metrics = trainer.collect_rollouts(50)
        assert metrics['timesteps_collected'] == 200  # 50 * 4 envs
        
    def test_trainer_cleanup_on_error(self):
        """Test proper cleanup when training fails."""
        envs = [gym.make('CartPole-v1') for _ in range(2)]
        agent = PPOAgent(observation_dim=4, action_dim=2)
        config = PPOConfig(num_workers=2)
        
        trainer = PPOTrainer(agent, envs, config, use_tensorboard=False)
        
        # Mock learn to fail
        with patch.object(trainer, 'learn', side_effect=RuntimeError("Training failed")):
            with pytest.raises(RuntimeError):
                trainer.train(total_timesteps=1000, rollout_length=100)
                
        # Executor should be shut down
        assert trainer.executor._shutdown


if __name__ == '__main__':
    # Run basic tests
    print("Running async PPO tests...")
    
    # Test 1: Thread safety
    print("\n1. Testing thread safety...")
    test_thread_safe = TestThreadSafeComponents()
    test_thread_safe.test_running_mean_std_thread_safety()
    test_thread_safe.test_rollout_buffer_thread_safety()
    print("✓ Thread safety tests passed")
    
    # Test 2: Performance comparison
    print("\n2. Testing async performance...")
    test_perf = TestAsyncPPOTrainer()
    test_perf.test_performance_vs_sequential()
    
    # Test 3: Error recovery
    print("\n3. Testing error recovery...")
    test_errors = TestErrorHandling()
    test_errors.test_env_failure_recovery()
    print("✓ Error recovery tests passed")
    
    print("\nAll tests completed!")