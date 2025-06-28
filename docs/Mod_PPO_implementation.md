# Modular PPO Implementation for JanusAI V2

A production-ready, modular implementation of Proximal Policy Optimization (PPO) designed for scalability, maintainability, and research flexibility.

## 📁 Project Structure

```
janus/training/ppo/
├── config.py           # Configuration management with YAML support
├── buffer.py           # Thread-safe rollout buffer
├── collector.py        # Asynchronous environment interaction
├── normalization.py    # Running statistics for reward normalization
├── logging_utils.py    # Logging backends (TensorBoard, W&B, CSV)
├── trainer.py          # Main PPO training orchestration
└── main.py            # Training entry point

configs/
├── demo_config.yaml              # PPO hyperparameters
├── actor_network_config.yaml     # Actor network architecture
└── critic_network_config.yaml    # Critic network architecture

tests/
└── test_ppo_integration.py      # Integration tests
```

## 🚀 Features

### Core Features
- **Modular Design**: Clean separation of concerns for easy maintenance
- **Thread-Safe Components**: All shared components use proper locking
- **Asynchronous Collection**: Parallel environment stepping with ThreadPoolExecutor
- **Flexible Configuration**: YAML-based configs with environment variable support
- **Production Ready**: Comprehensive error handling and recovery

### Training Features
- **Mixed Precision Training**: Automatic mixed precision (AMP) support
- **Gradient Accumulation**: For large effective batch sizes
- **Hyperparameter Scheduling**: Linear/cosine schedules for LR, clip, entropy
- **Early Stopping**: KL-divergence based early stopping
- **Best Model Tracking**: Automatically saves best model during training

### Logging & Monitoring
- **Multiple Backends**: TensorBoard, Weights & Biases, CSV
- **Comprehensive Metrics**: Policy loss, value loss, entropy, KL divergence, explained variance
- **Real-time Statistics**: Episode rewards, lengths, training speed

### Advanced Features
- **Reward Normalization**: Running mean/std normalization
- **Advantage Normalization**: Per-batch advantage standardization
- **Evaluation Support**: Periodic evaluation during training
- **Checkpointing**: Save/load complete training state

## 🔧 Installation

```bash
# Install dependencies
pip install torch numpy gymnasium tensorboard
pip install wandb  # Optional, for W&B logging

# Clone the repository
git clone <repository-url>
cd janus
```

## 📖 Quick Start

### 1. Create Configuration Files

**demo_config.yaml**:
```yaml
learning_rate: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_epsilon: 0.2
n_epochs: 10
batch_size: 64
normalize_rewards: true
lr_schedule: cosine
num_workers: 4
```

**actor_network_config.yaml**:
```yaml
layer_sizes: [256, 256]
activation: tanh
initialization: orthogonal
```

### 2. Run Training

```bash
# Basic training
python -m janus.training.ppo.main --config demo_config.yaml

# With custom parameters
python -m janus.training.ppo.main \
    --config demo_config.yaml \
    --total-timesteps 1000000 \
    --num-envs 8 \
    --tensorboard \
    --eval

# Resume from checkpoint
python -m janus.training.ppo.main \
    --config demo_config.yaml \
    --resume checkpoints/best_model.pt
```

### 3. Monitor Training

```bash
# TensorBoard
tensorboard --logdir runs/

# View CSV logs
tail -f checkpoints/ppo_experiment/metrics.csv
```

## 🎯 Usage Examples

### Basic Training Script

```python
from janus.training.ppo import PPOConfig, PPOTrainer
from janus.agents.ppo_agent import PPOAgent, NetworkConfig
from janus.envs.symbolic_regression import SymbolicRegressionEnv

# Load configuration
config = PPOConfig.from_yaml("demo_config.yaml")

# Create environments
envs = [SymbolicRegressionEnv() for _ in range(4)]
eval_env = SymbolicRegressionEnv()

# Create agent
agent = PPOAgent(
    observation_dim=envs[0].observation_space.shape[0],
    action_dim=envs[0].action_space.n,
    actor_config=NetworkConfig(layer_sizes=[256, 256]),
    critic_config=NetworkConfig(layer_sizes=[256, 256])
)

# Create trainer
trainer = PPOTrainer(
    agent=agent,
    envs=envs,
    config=config,
    experiment_name="my_experiment",
    use_tensorboard=True,
    eval_env=eval_env
)

# Train
trainer.train(total_timesteps=1_000_000, rollout_length=2048)
```

### Custom Environment

```python
import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-1, 1, shape=(10,))
        self.action_space = spaces.Discrete(4)
        
    def reset(self):
        return np.random.randn(10), {}
        
    def step(self, action):
        obs = np.random.randn(10)
        reward = np.random.randn()
        done = np.random.random() < 0.01
        return obs, reward, done, False, {}

# Use with PPO
envs = [CustomEnv() for _ in range(8)]
# ... rest of training setup
```

### Hyperparameter Tuning

```python
# Create sweep configuration
sweep_configs = [
    {"learning_rate": 1e-4, "entropy_coef": 0.01},
    {"learning_rate": 3e-4, "entropy_coef": 0.01},
    {"learning_rate": 3e-4, "entropy_coef": 0.001},
]

for i, sweep_config in enumerate(sweep_configs):
    config = PPOConfig(**sweep_config)
    trainer = PPOTrainer(
        agent=agent,
        envs=envs,
        config=config,
        experiment_name=f"sweep_{i}"
    )
    trainer.train(total_timesteps=100_000)
```

## 🔍 Configuration Options

### PPO Hyperparameters
- `learning_rate`: Initial learning rate (default: 3e-4)
- `gamma`: Discount factor (default: 0.99)
- `gae_lambda`: GAE lambda parameter (default: 0.95)
- `clip_epsilon`: PPO clipping parameter (default: 0.2)
- `n_epochs`: Training epochs per update (default: 10)
- `batch_size`: Minibatch size (default: 64)

### Scheduling Options
- `lr_schedule`: Learning rate schedule (constant/linear/cosine)
- `clip_schedule`: Clip epsilon schedule
- `entropy_schedule`: Entropy coefficient schedule

### Training Options
- `normalize_rewards`: Enable reward normalization
- `normalize_advantages`: Enable advantage normalization
- `use_mixed_precision`: Enable AMP for faster training
- `gradient_accumulation_steps`: Steps to accumulate gradients

### System Options
- `device`: Training device (auto/cpu/cuda)
- `num_workers`: Parallel environments
- `use_subprocess_envs`: Use subprocess for environments

## 🧪 Testing

```bash
# Run integration tests
pytest tests/test_ppo_integration.py -v

# Quick test
python tests/test_ppo_integration.py
```

## 📊 Monitoring Metrics

The trainer logs comprehensive metrics:

- **Training Metrics**:
  - `mean_episode_reward`: Average episode reward
  - `policy_loss`: PPO policy loss
  - `value_loss`: Value function loss
  - `entropy`: Policy entropy
  - `kl_divergence`: KL divergence between old and new policy
  - `clip_fraction`: Fraction of clipped ratios
  - `explained_variance`: Explained variance of value function
  - `steps_per_second`: Training throughput

- **Evaluation Metrics**:
  - `eval_reward`: Mean evaluation episode reward
  - `eval_reward_std`: Standard deviation of eval rewards
  - `eval_length`: Mean evaluation episode length

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce `batch_size` or `rollout_length`
   - Disable `use_mixed_precision` if using old GPU
   - Reduce number of environments

2. **Slow Training**:
   - Increase `num_workers` for CPU-bound environments
   - Enable `use_mixed_precision` on compatible GPUs
   - Use `use_subprocess_envs` for heavy environments

3. **Unstable Training**:
   - Reduce `learning_rate`
   - Increase `n_epochs`
   - Enable `normalize_rewards`
   - Adjust `target_kl` for early stopping

4. **Environment Errors**:
   - Set `fail_on_env_error: false` for graceful recovery
   - Increase `max_env_retries`
   - Check environment implementation

## 🚀 Performance Tips

1. **Use Mixed Precision**: Enable `use_mixed_precision` for 2x speedup on modern GPUs
2. **Optimize Batch Size**: Larger batches are more GPU-efficient
3. **Parallel Environments**: Use multiple environments (`num_workers`)
4. **Gradient Accumulation**: Simulate larger batches without memory increase
5. **Profile Your Code**: Use PyTorch profiler to find bottlenecks

## 📝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This implementation is part of the JanusAI V2 framework.
