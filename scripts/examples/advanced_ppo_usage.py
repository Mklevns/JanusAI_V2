# examples/advanced_ppo_usage.py
"""
Advanced usage examples for the production PPO trainer.

Demonstrates:
1. YAML configuration management
2. Hyperparameter sweeps
3. Async performance comparison
4. Distributed training setup
5. Custom environment wrappers
6. Advanced monitoring
"""
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
import multiprocessing as mp

import numpy as np
import torch
import gym
from gym import spaces
import yaml

from janus.training.ppo_trainer import PPOTrainer, PPOConfig, create_ppo_trainer_from_yaml
from janus.agents.ppo_agent import PPOAgent


# =============================================================================
# Example 1: YAML Configuration with Environment Variables
# =============================================================================

def create_experiment_config():
    """Create a comprehensive YAML configuration for experiments."""
    
    config_yaml = """
# PPO Hyperparameters
learning_rate: ${LEARNING_RATE:3e-4}
gamma: 0.99
gae_lambda: 0.95
clip_epsilon: ${CLIP_EPSILON:0.2}
value_coef: 0.5
entropy_coef: ${ENTROPY_COEF:0.01}
n_epochs: 10
batch_size: ${BATCH_SIZE:64}

# Optimization
max_grad_norm: 0.5
use_mixed_precision: true
gradient_accumulation_steps: ${GRAD_ACCUM:1}

# Scheduling
lr_schedule: cosine
clip_schedule: linear
entropy_schedule: linear
lr_end: 1e-5
clip_end: 0.1
entropy_end: 0.001

# Early Stopping
target_kl: 0.02

# Normalization
normalize_advantages: true
normalize_rewards: ${NORMALIZE_REWARDS:false}

# Execution
device: auto
num_workers: ${NUM_WORKERS:4}
use_subprocess_envs: true

# Error Handling
fail_on_env_error: false
max_env_retries: 3

# Logging
log_interval: 10
save_interval: 100
eval_interval: 500

# Experiment Tags
experiment_tags:
  project: janus
  algorithm: ppo
  version: ${VERSION:1.0}
  hardware: ${HARDWARE:gpu}
"""
    
    # Save to file
    with open("experiment_config.yaml", "w") as f:
        f.write(config_yaml)
    
    print("Created experiment_config.yaml with environment variable support")
    
    # Example: Set environment variables and load
    import os
    os.environ['LEARNING_RATE'] = '5e-4'
    os.environ['BATCH_SIZE'] = '128'
    os.environ['NUM_WORKERS'] = '8'
    
    # Load config
    config = PPOConfig.from_yaml("experiment_config.yaml")
    print(f"Loaded config: LR={config.learning_rate}, Batch={config.batch_size}, Workers={config.num_workers}")
    
    return config


# =============================================================================
# Example 2: Hyperparameter Sweep
# =============================================================================

def hyperparameter_sweep():
    """Run a hyperparameter sweep with different configurations."""
    
    base_config = {
        'gamma': 0.99,
        'n_epochs': 10,
        'normalize_rewards': True,
        'use_mixed_precision': torch.cuda.is_available(),
    }
    
    # Define sweep parameters
    sweep_configs = [
        {'learning_rate': 1e-4, 'entropy_coef': 0.01, 'clip_epsilon': 0.2},
        {'learning_rate': 3e-4, 'entropy_coef': 0.01, 'clip_epsilon': 0.2},
        {'learning_rate': 5e-4, 'entropy_coef': 0.01, 'clip_epsilon': 0.2},
        {'learning_rate': 3e-4, 'entropy_coef': 0.001, 'clip_epsilon': 0.2},
        {'learning_rate': 3e-4, 'entropy_coef': 0.1, 'clip_epsilon': 0.2},
        {'learning_rate': 3e-4, 'entropy_coef': 0.01, 'clip_epsilon': 0.1},
        {'learning_rate': 3e-4, 'entropy_coef': 0.01, 'clip_epsilon': 0.3},
    ]
    
    results = []
    
    for i, sweep_params in enumerate(sweep_configs):
        print(f"\n=== Sweep {i+1}/{len(sweep_configs)} ===")
        print(f"Parameters: {sweep_params}")
        
        # Create config
        config_dict = {**base_config, **sweep_params}
        config = PPOConfig(**config_dict)
        
        # Create experiment name
        exp_name = f"sweep_lr{sweep_params['learning_rate']}_ent{sweep_params['entropy_coef']}_clip{sweep_params['clip_epsilon']}"
        
        # Run short training
        try:
            # Setup environment and agent
            env = gym.make('CartPole-v1')
            agent = PPOAgent(
                observation_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n
            )
            
            # Create trainer
            trainer = PPOTrainer(
                agent=agent,
                env=[env],
                config=config,
                experiment_name=exp_name,
                use_tensorboard=False,
                use_wandb=False
            )
            
            # Train for a short time
            trainer.train(total_timesteps=10000, rollout_length=512)
            
            # Record results
            results.append({
                'config': sweep_params,
                'final_reward': trainer.best_mean_reward,
                'experiment_name': exp_name
            })
            
        except Exception as e:
            print(f"Sweep {i+1} failed: {e}")
            results.append({
                'config': sweep_params,
                'final_reward': -float('inf'),
                'error': str(e)
            })
    
    # Report results
    print("\n=== Sweep Results ===")
    results.sort(key=lambda x: x['final_reward'], reverse=True)
    for i, result in enumerate(results):
        print(f"{i+1}. Reward: {result['final_reward']:.2f} - Config: {result['config']}")
    
    # Save results
    with open("sweep_results.yaml", "w") as f:
        yaml.dump(results, f)
    
    return results


# =============================================================================
# Example 3: Async Performance Comparison
# =============================================================================

def benchmark_async_performance():
    """Compare performance of different async strategies."""
    
    import time
    import matplotlib.pyplot as plt
    
    # Test configurations
    test_configs = [
        {'name': 'Sequential', 'num_workers': 1, 'use_subprocess': False},
        {'name': 'ThreadPool-2', 'num_workers': 2, 'use_subprocess': False},
        {'name': 'ThreadPool-4', 'num_workers': 4, 'use_subprocess': False},
        {'name': 'ThreadPool-8', 'num_workers': 8, 'use_subprocess': False},
    ]
    
    # Add subprocess configs if available
    try:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        test_configs.extend([
            {'name': 'Subprocess-4', 'num_workers': 4, 'use_subprocess': True},
            {'name': 'Subprocess-8', 'num_workers': 8, 'use_subprocess': True},
        ])
    except ImportError:
        print("SubprocVecEnv not available, skipping subprocess tests")
    
    results = []
    n_envs = 8
    rollout_length = 512
    
    for test_config in test_configs:
        print(f"\nTesting {test_config['name']}...")
        
        # Create config
        config = PPOConfig(
            num_workers=test_config['num_workers'],
            use_subprocess_envs=test_config['use_subprocess'],
            log_interval=1000  # Reduce logging
        )
        
        # Create environments
        envs = [gym.make('CartPole-v1') for _ in range(n_envs)]
        
        # Create agent
        agent = PPOAgent(
            observation_dim=envs[0].observation_space.shape[0],
            action_dim=envs[0].action_space.n
        )
        
        # Create trainer
        trainer = PPOTrainer(
            agent=agent,
            env=envs,
            config=config,
            experiment_name=f"async_bench_{test_config['name']}",
            use_tensorboard=False,
            use_wandb=False
        )
        
        # Initialize buffer
        trainer.buffer = trainer.buffer or RolloutBuffer(
            buffer_size=rollout_length,
            obs_shape=trainer.obs_shape,
            action_shape=trainer.action_shape if trainer.continuous_actions else (),
            device=trainer.device,
            n_envs=trainer.n_envs
        )
        
        # Measure collection time
        start_time = time.time()
        collection_metrics = trainer.collect_rollouts(rollout_length)
        collection_time = time.time() - start_time
        
        steps_per_second = (rollout_length * n_envs) / collection_time
        
        results.append({
            'name': test_config['name'],
            'steps_per_second': steps_per_second,
            'collection_time': collection_time,
            'config': test_config
        })
        
        print(f"{test_config['name']}: {steps_per_second:.1f} steps/sec")
        
        # Cleanup
        trainer.executor.shutdown() if trainer.executor else None
    
    # Plot results
    plt.figure(figsize=(10, 6))
    names = [r['name'] for r in results]
    speeds = [r['steps_per_second'] for r in results]
    
    bars = plt.bar(names, speeds)
    plt.xlabel('Configuration')
    plt.ylabel('Steps per Second')
    plt.title('Async Rollout Collection Performance')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, speed in zip(bars, speeds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{speed:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('async_performance.png')
    print("\nPerformance plot saved to async_performance.png")
    
    return results


# =============================================================================
# Example 4: Custom Environment Wrapper for Monitoring
# =============================================================================

class DetailedMonitorWrapper(gym.Wrapper):
    """Custom wrapper that tracks detailed episode statistics."""
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_reward = 0
        self.current_length = 0
        self.action_counts = {}
        
    def reset(self, **kwargs):
        if self.current_length > 0:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
        
        self.current_reward = 0
        self.current_length = 0
        self.action_counts.clear()
        
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        self.current_reward += reward
        self.current_length += 1
        
        # Track action distribution
        action_key = str(action)
        self.action_counts[action_key] = self.action_counts.get(action_key, 0) + 1
        
        # Add custom info
        info['episode_reward'] = self.current_reward
        info['episode_length'] = self.current_length
        info['action_distribution'] = dict(self.action_counts)
        
        return obs, reward, done, truncated, info
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics."""
        if not self.episode_rewards:
            return {}
            
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'total_episodes': len(self.episode_rewards),
        }


# =============================================================================
# Example 5: Distributed Training Setup (Ray)
# =============================================================================

def setup_distributed_training():
    """Example setup for distributed training using Ray."""
    
    try:
        import ray
        from ray import tune
        from ray.tune.integration.wandb import WandbLoggerCallback
    except ImportError:
        print("Ray not installed. Install with: pip install ray[tune]")
        return
    
    def train_ppo(config):
        """Training function for Ray Tune."""
        # Create PPO config from Ray config
        ppo_config = PPOConfig(
            learning_rate=config['learning_rate'],
            clip_epsilon=config['clip_epsilon'],
            entropy_coef=config['entropy_coef'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
        )
        
        # Create environment
        env = gym.make('CartPole-v1')
        
        # Create agent
        agent = PPOAgent(
            observation_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n
        )
        
        # Create trainer
        trainer = PPOTrainer(
            agent=agent,
            env=[env],
            config=ppo_config,
            experiment_name=f"ray_ppo_{config['trial_id']}",
            use_tensorboard=False,
            use_wandb=False
        )
        
        # Training loop with reporting
        for i in range(config['num_iterations']):
            trainer.collect_rollouts(512)
            metrics = trainer.learn(
                current_clip_epsilon=ppo_config.clip_epsilon,
                current_entropy_coef=ppo_config.entropy_coef
            )
            
            # Report to Ray Tune
            tune.report(
                iteration=i,
                mean_reward=trainer.collector.get_statistics()['mean_episode_reward'],
                policy_loss=metrics['policy_loss'],
                value_loss=metrics['value_loss']
            )
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Define search space
    search_space = {
        'learning_rate': tune.loguniform(1e-5, 1e-3),
        'clip_epsilon': tune.uniform(0.1, 0.3),
        'entropy_coef': tune.loguniform(1e-4, 1e-1),
        'batch_size': tune.choice([32, 64, 128]),
        'n_epochs': tune.choice([5, 10, 15]),
        'num_iterations': 100,
        'trial_id': tune.trial_id(),
    }
    
    # Run hyperparameter search
    analysis = tune.run(
        train_ppo,
        config=search_space,
        num_samples=10,
        metric='mean_reward',
        mode='max',
        resources_per_trial={'cpu': 2, 'gpu': 0.5 if torch.cuda.is_available() else 0},
        callbacks=[WandbLoggerCallback(project='ray-ppo')] if WANDB_AVAILABLE else [],
    )
    
    # Get best config
    best_config = analysis.get_best_config(metric='mean_reward', mode='max')
    print(f"\nBest configuration: {best_config}")
    
    # Shutdown Ray
    ray.shutdown()
    
    return analysis


# =============================================================================
# Example 6: Advanced Monitoring Dashboard
# =============================================================================

def create_monitoring_dashboard():
    """Create a real-time monitoring dashboard using Streamlit."""
    
    dashboard_code = '''
# dashboard.py
"""
Real-time PPO training dashboard.
Run with: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
import numpy as np

st.set_page_config(page_title="PPO Training Dashboard", layout="wide")
st.title("🚀 PPO Training Dashboard")

# Sidebar configuration
st.sidebar.header("Configuration")
experiment_path = st.sidebar.text_input("Experiment Path", "checkpoints/ppo_experiment")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 2)
show_raw_data = st.sidebar.checkbox("Show Raw Data", False)

# Create layout
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Step", "0", "+0")
with col2:
    st.metric("Best Reward", "0.0", "+0.0")
with col3:
    st.metric("Learning Rate", "0.0001", "")

# Main plots
plot_container = st.container()
with plot_container:
    # Reward plot
    st.subheader("Episode Rewards")
    reward_plot = st.empty()
    
    # Loss plots
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Policy Loss")
        policy_loss_plot = st.empty()
    with col2:
        st.subheader("Value Loss")
        value_loss_plot = st.empty()
    
    # Additional metrics
    st.subheader("Training Metrics")
    metrics_plot = st.empty()

# Load and display data
def load_metrics(path):
    """Load metrics from CSV file."""
    csv_path = Path(path) / "metrics.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()

# Auto-refresh loop
placeholder = st.empty()
while True:
    try:
        # Load data
        df = load_metrics(experiment_path)
        
        if not df.empty:
            # Update metrics
            current_step = df['global_step'].iloc[-1]
            best_reward = df['mean_reward'].max()
            current_lr = df['learning_rate'].iloc[-1]
            
            col1.metric("Current Step", f"{current_step:,}", f"+{df['global_step'].diff().iloc[-1]:,.0f}")
            col2.metric("Best Reward", f"{best_reward:.2f}", f"+{df['mean_reward'].diff().iloc[-1]:.2f}")
            col3.metric("Learning Rate", f"{current_lr:.6f}", "")
            
            # Update plots
            # Reward plot
            fig_reward = px.line(df, x='global_step', y='mean_reward', 
                                title='Episode Reward Over Time')
            fig_reward.add_hline(y=best_reward, line_dash="dash", 
                                annotation_text=f"Best: {best_reward:.2f}")
            reward_plot.plotly_chart(fig_reward, use_container_width=True)
            
            # Loss plots
            fig_policy = px.line(df, x='global_step', y='policy_loss')
            policy_loss_plot.plotly_chart(fig_policy, use_container_width=True)
            
            fig_value = px.line(df, x='global_step', y='value_loss')
            value_loss_plot.plotly_chart(fig_value, use_container_width=True)
            
            # Combined metrics
            fig_metrics = go.Figure()
            fig_metrics.add_trace(go.Scatter(x=df['global_step'], y=df['entropy'], 
                                            name='Entropy', yaxis='y'))
            fig_metrics.add_trace(go.Scatter(x=df['global_step'], y=df['kl_divergence'], 
                                            name='KL Divergence', yaxis='y2'))
            fig_metrics.update_layout(
                yaxis=dict(title='Entropy'),
                yaxis2=dict(title='KL Divergence', overlaying='y', side='right')
            )
            metrics_plot.plotly_chart(fig_metrics, use_container_width=True)
            
            # Show raw data if requested
            if show_raw_data:
                st.subheader("Raw Data")
                st.dataframe(df.tail(20))
        else:
            st.warning("No data found. Check experiment path.")
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
    
    # Wait before refresh
    time.sleep(refresh_rate)
    placeholder.empty()
'''
    
    # Save dashboard code
    with open("dashboard.py", "w") as f:
        f.write(dashboard_code)
    
    print("Created dashboard.py - Run with: streamlit run dashboard.py")
    
    return dashboard_code


# =============================================================================
# Main Examples Runner
# =============================================================================

def run_all_examples():
    """Run all example demonstrations."""
    
    print("=" * 80)
    print("PPO Advanced Usage Examples")
    print("=" * 80)
    
    # Example 1: YAML Configuration
    print("\n1. YAML Configuration Management")
    print("-" * 40)
    config = create_experiment_config()
    
    # Example 2: Hyperparameter Sweep
    print("\n2. Hyperparameter Sweep")
    print("-" * 40)
    # sweep_results = hyperparameter_sweep()  # Uncomment to run
    print("Skipping sweep (uncomment to run)")
    
    # Example 3: Async Performance
    print("\n3. Async Performance Benchmark")
    print("-" * 40)
    async_results = benchmark_async_performance()
    
    # Example 4: Custom Wrappers
    print("\n4. Custom Environment Wrapper")
    print("-" * 40)
    env = DetailedMonitorWrapper(gym.make('CartPole-v1'))
    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break
    print(f"Episode stats: {env.get_statistics()}")
    
    # Example 5: Distributed Training
    print("\n5. Distributed Training Setup")
    print("-" * 40)
    # setup_distributed_training()  # Uncomment if Ray is installed
    print("Skipping distributed setup (requires Ray)")
    
    # Example 6: Monitoring Dashboard
    print("\n6. Monitoring Dashboard")
    print("-" * 40)
    create_monitoring_dashboard()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    run_all_examples()