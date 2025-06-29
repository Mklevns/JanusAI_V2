# World Model PPO Implementation

This implementation adds World Model capabilities to the JanusAI V2 PPO trainer, following the approach from the "World Models" paper (Ha & Schmidhuber, 2018). The World Model allows agents to learn from both real environment interactions and imagined rollouts in a learned model of the environment.

## 🏗️ Architecture

The World Model consists of three main components:

### 1. Vision (V) - Variational Autoencoder (VAE)

- **Location**: `janus/agents/components/vae.py`
- **Purpose**: Compresses high-dimensional observations into compact latent representations
- **Key Features**:
  - β-VAE implementation for disentangled representations
  - Configurable architecture with layer normalization
  - Supports both training and deterministic inference modes

### 2. Memory (M) - Mixture Density Network RNN (MDN-RNN)

- **Location**: `janus/agents/components/mdn_rnn.py`
- **Purpose**: Models the dynamics of the environment in latent space
- **Key Features**:
  - LSTM-based sequence modeling
  - Mixture of Gaussians output for probabilistic predictions
  - Temperature-controlled sampling

### 3. Controller (C) - Modified PPO Agent

- **Purpose**: Makes decisions based on latent states and RNN hidden states
- **Key Features**:
  - Augmented input: latent state + RNN hidden state
  - Seamless integration with existing PPO implementation

## 🚀 Quick Start

### 1. Basic Usage

```python
from janus.agents.ppo_agent import PPOAgent, NetworkConfig
from janus.training.ppo.config import PPOConfig
from janus.training.ppo.world_model_trainer import WorldModelPPOTrainer, WorldModelConfig

# Create configurations
ppo_config = PPOConfig.from_yaml("configs/world_model_ppo.yaml")
wm_config = WorldModelConfig(
    vae_latent_dim=32,
    imagination_ratio=0.5,  # 50% real, 50% imagined experience
    imagination_horizon=15   # Steps to imagine ahead
)

# Create agent
agent = PPOAgent(
    observation_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    actor_config=NetworkConfig(layer_sizes=[256, 128]),
    critic_config=NetworkConfig(layer_sizes=[256, 128])
)

# Create trainer
trainer = WorldModelPPOTrainer(
    agent=agent,
    envs=envs,
    config=ppo_config,
    world_model_config=wm_config,
    experiment_name="my_world_model_experiment"
)

# Train
trainer.train(total_timesteps=1_000_000)
```

### 2. Using the Training Script

```bash
# Train on CartPole with default settings
python scripts/train_world_model_ppo.py --env CartPole-v1 --total-timesteps 1000000

# Train with custom config and GPU
python scripts/train_world_model_ppo.py \
    --config configs/world_model_ppo.yaml \
    --env LunarLander-v2 \
    --device cuda \
    --num-envs 8 \
    --tensorboard

# Resume from checkpoint
python scripts/train_world_model_ppo.py \
    --resume checkpoints/world_model_ppo/checkpoint_50000.pt
```

## 🛠️ Configuration

The World Model configuration is part of the main YAML config file:

```yaml
# configs/world_model_ppo.yaml

world_model:
  # VAE configuration
  vae:
    latent_dim: 32              # Size of latent representation
    hidden_dims: [256, 128]     # Hidden layers for encoder/decoder
    beta: 1.0                   # β-VAE weight (higher = more disentangled)
    learning_rate: 0.001

  # MDN-RNN configuration
  mdn_rnn:
    hidden_dim: 256            # LSTM hidden size
    num_mixtures: 5            # Number of Gaussian mixtures
    temperature: 1.0           # Sampling temperature
    learning_rate: 0.001

  # Training configuration
  training:
    pretrain_epochs: 10        # Epochs to pretrain world model
    imagination_ratio: 0.5     # Fraction of imagined experience
    imagination_horizon: 15    # Steps to imagine into future

  # Data collection
  data_collection:
    random_collection_steps: 10000  # Initial random exploration
```

## 🔬 How It Works

### Training Process

1. **Random Data Collection**: Initially collects experience using a random policy
2. **World Model Pretraining**:
   - VAE learns to encode/decode observations
   - MDN-RNN learns to predict future latent states
3. **Mixed Training**:
   - Collects real experience from the environment
   - Generates imagined experience using the world model
   - Trains PPO on combined real + imagined data

### Key Benefits

- **Sample Efficiency**: Learn from imagined experience without environment interaction
- **Planning**: Agent can "think ahead" using the world model
- **Exploration**: Can safely explore in imagination before trying in reality
- **Transfer**: World model can potentially transfer to similar tasks

## 📊 Monitoring Training

The trainer logs additional metrics for the World Model:

- `vae/reconstruction_loss`: How well the VAE reconstructs observations
- `vae/kl_divergence`: Regularization term for the VAE
- `mdn/nll`: Prediction accuracy of the MDN-RNN
- `imagination/ratio`: Actual ratio of imagined vs real experience
- `imagination/horizon`: Average imagination rollout length

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all World Model tests
pytest tests/test_world_model.py -v

# Run specific component tests
pytest tests/test_world_model.py::TestVAE -v
pytest tests/test_world_model.py::TestMDNRNN -v
pytest tests/test_world_model.py::TestWorldModelPPOTrainer -v

# Quick integration test
python tests/test_world_model.py
```

## 🎯 Best Practices

1. **Pretraining Data**: Collect diverse random experience for good world model coverage
2. **Latent Dimension**: Start with 32-64, adjust based on environment complexity
3. **Imagination Ratio**: Start with 0.3-0.5, increase as world model improves
4. **Horizon Length**: Shorter horizons (10-20) are more accurate
5. **Monitoring**: Watch VAE reconstruction quality - poor reconstruction hurts performance

## 🔧 Extending the Implementation

### Custom Environment Preprocessing

```python
class CustomVAE(VariationalAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom preprocessing layers
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
```

### Planning with the World Model

```python
def plan_with_world_model(trainer, start_obs, num_trajectories=100):
    """Use the world model for planning."""
    best_trajectory = None
    best_return = -float('inf')

    for _ in range(num_trajectories):
        trajectory = trainer.imagine_rollout(start_obs, horizon=20)
        returns = trajectory['values'].sum(dim=1)

        if returns.max() > best_return:
            best_return = returns.max()
            best_trajectory = trajectory

    return best_trajectory
```

## 📚 References

- [World Models Paper](https://arxiv.org/abs/1803.10122)
- [β-VAE Paper](https://arxiv.org/abs/1606.05579)
- [MDN Paper](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf)

## 🐛 Troubleshooting

### Common Issues

1. **VAE not learning**:
   - Check beta value (try 0.1-1.0)
   - Ensure observations are normalized
   - Try different architectures

2. **MDN-RNN predictions unstable**:
   - Reduce number of mixtures
   - Clip log_sigma values
   - Use gradient clipping

3. **Agent performance degrades with imagination**:
   - Reduce imagination ratio
   - Ensure world model is well-trained
   - Check for distribution shift

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger('janus.training.ppo.world_model_trainer').setLevel(logging.DEBUG)
```
