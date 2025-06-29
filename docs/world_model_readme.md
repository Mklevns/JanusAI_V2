# World Model Implementation for JanusAI_V2

This implementation adds a World Model capability to your JanusAI_V2 project, following the architecture from the "World Models" paper by Ha and Schmidhuber. The World Model enables agents to learn compressed representations of their environment and imagine future trajectories for improved planning and sample efficiency.

## Architecture Overview

The World Model consists of three main components:

1. **Vision (V)**: A Variational Autoencoder (VAE) that compresses high-dimensional observations into low-dimensional latent vectors
2. **Memory (M)**: A Mixture Density Network RNN (MDN-RNN) that predicts future latent states
3. **Controller (C)**: A PPO agent that acts based on the latent representation and RNN hidden state

```
   [Environment]
         |
    (Observation)
         |
    [    VAE    ] (V)
         |
    (Latent z_t)
         |
         |---------> [PPO Controller] (C)
         |                   |
    [ MDN-RNN ] (M) <--- (Action)
         |
    (Hidden h_t) -----------|
```

## Installation

No additional dependencies are required beyond your existing setup. The implementation uses PyTorch and integrates seamlessly with your current PPO infrastructure.

## Quick Start

### 1. Pre-train the World Model Components

First, collect random exploration data and train the VAE and MDN-RNN:

```python
from janus.training.world_model.train_world_model import WorldModelTrainer, WorldModelTrainingConfig
import gymnasium as gym

# Create environment
env = gym.make('CarRacing-v2')  # Works well with visual environments

# Configure training
config = WorldModelTrainingConfig(
    num_episodes=1000,
    vae_epochs=30,
    mdn_epochs=50,
    device='cuda'
)

# Train world model
trainer = WorldModelTrainer(env, config)
vae, mdn_rnn = trainer.train_world_model()
```

### 2. Create a World Model Agent

```python
from janus.agents.world_model_agent import WorldModelAgent

# Create agent with trained components
agent = WorldModelAgent(
    observation_shape=(3, 64, 64),  # CarRacing observation shape
    action_dim=env.action_space.shape[0],
    continuous_actions=True,
    device='cuda'
)

# Load pre-trained components
agent.vae.load_state_dict(vae.state_dict())
agent.mdn_rnn.load_state_dict(mdn_rnn.state_dict())
```

### 3. Train the Controller with PPO

Use the extended PPO trainer that can learn from both real and imagined experience:

```python
from janus.examples.train_ppo_with_world_model import WorldModelPPOTrainer
from janus.training.ppo.config import PPOConfig

# Create PPO config
ppo_config = PPOConfig(
    learning_rate=3e-4,
    n_epochs=10,
    batch_size=64
)

# Create trainer with imagination
trainer = WorldModelPPOTrainer(
    world_model_agent=agent,
    envs=[env],  # Can use multiple environments
    config=ppo_config,
    imagination_ratio=0.5,  # 50% imagined data
    imagination_horizon=50,  # Look 50 steps into the future
    use_tensorboard=True
)

# Train
trainer.train(total_timesteps=1_000_000, rollout_length=2048)
```

## Key Features

### 1. Modular Design
Each component (VAE, MDN-RNN, Controller) can be trained and used independently:

```python
# Use just the VAE for observation encoding
from janus.agents.components.vae import VariationalAutoencoder, VAEConfig

vae = VariationalAutoencoder(VAEConfig(
    input_channels=3,
    latent_dim=32
))

# Encode observations
latent = vae.get_latent(observation)
```

### 2. Imagination-Based Planning
The agent can imagine future trajectories without environment interaction:

```python
# Imagine a trajectory from current observation
trajectory = agent.imagine_trajectory(
    initial_obs=current_observation,
    horizon=50,
    temperature=1.0  # Control exploration
)

# trajectory contains:
# - 'latents': predicted latent states
# - 'actions': imagined actions
```

### 3. Flexible Training
You can adjust the balance between real and imagined experience:

```python
# More imagination for sample efficiency
trainer = WorldModelPPOTrainer(
    imagination_ratio=0.8,  # 80% imagined data
    ...
)

# More real data for accuracy
trainer = WorldModelPPOTrainer(
    imagination_ratio=0.2,  # 20% imagined data
    ...
)
```

## Integration with Existing Code

The World Model implementation integrates seamlessly with your existing infrastructure:

### Using with Your PPOAgent
The `WorldModelAgent` wraps your existing `PPOAgent` as the controller:

```python
from janus.agents.ppo_agent import NetworkConfig

# Your existing network config works
controller_config = NetworkConfig(
    layer_sizes=[256, 256],
    activation="tanh",
    use_layer_norm=True
)

# Pass it to WorldModelAgent
agent = WorldModelAgent(
    controller_config=controller_config,
    ...
)
```

### Using with Your Environments
Works with any Gymnasium-compatible environment:

```python
# Your symbolic regression environment
from janus.envs.symbolic_regression import SymbolicRegressionEnv
env = SymbolicRegressionEnv()

# Train world model on it
trainer = WorldModelTrainer(env, config)
```

## Configuration Options

### VAE Configuration
```python
VAEConfig(
    input_channels=3,        # Number of input channels
    input_height=64,         # Input image height
    input_width=64,          # Input image width
    latent_dim=32,          # Size of latent representation
    hidden_channels=[32, 64, 128, 256],  # Conv layer channels
    beta=1.0                # KL loss weight
)
```

### MDN-RNN Configuration
```python
MDNRNNConfig(
    latent_dim=32,          # Must match VAE latent_dim
    action_dim=4,           # Environment action dimension
    hidden_dim=256,         # RNN hidden size
    num_mixtures=5,         # Number of Gaussian mixtures
    rnn_type="lstm",        # "lstm" or "gru"
    rnn_layers=1,           # Number of RNN layers
    temperature=1.0         # Sampling temperature
)
```

## Testing

Run the comprehensive test suite:

```bash
# Test all world model components
pytest tests/test_world_model.py -v

# Test VAE only
pytest tests/test_world_model.py::TestVAE -v

# Test MDN-RNN only
pytest tests/test_world_model.py::TestMDNRNN -v

# Test integration
pytest tests/test_world_model.py::TestIntegration -v
```

## Advanced Usage

### Custom Reward Models
For imagination-based training, you can implement custom reward models:

```python
class LearnedRewardModel(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2 + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, z_t, a_t, z_next):
        x = torch.cat([z_t, a_t, z_next], dim=-1)
        return self.net(x).squeeze(-1)
```

### Dreaming for Data Augmentation
Use the world model to generate synthetic training data:

```python
# Generate 1000 imagined trajectories
synthetic_data = []
for _ in range(1000):
    # Random starting observation
    start_obs = env.observation_space.sample()

    # Imagine trajectory
    trajectory = agent.imagine_trajectory(
        torch.from_numpy(start_obs),
        horizon=100
    )

    synthetic_data.append(trajectory)
```

### Multi-Task Learning
Train a single world model across multiple environments:

```python
# Train on multiple environments
envs = [
    gym.make('CarRacing-v2'),
    gym.make('LunarLander-v2'),
    gym.make('BipedalWalker-v3')
]

# Collect data from all environments
all_observations = []
all_actions = []
for env in envs:
    obs, actions = collect_data(env)
    all_observations.extend(obs)
    all_actions.extend(actions)

# Train shared world model
vae = train_vae(all_observations)
mdn_rnn = train_mdn_rnn(vae, all_observations, all_actions)
```

## Troubleshooting

### Memory Issues
If you run out of memory during training:
1. Reduce `vae_batch_size` or `mdn_batch_size` in the config
2. Reduce the `latent_dim` to compress observations more
3. Use gradient accumulation for larger effective batch sizes

### Poor Reconstruction Quality
If the VAE reconstructions are poor:
1. Increase the number of VAE training epochs
2. Increase the `hidden_channels` in VAEConfig
3. Adjust the `beta` parameter (lower for better reconstruction, higher for better latent space)

### Unstable MDN-RNN Training
If the MDN-RNN loss explodes:
1. Reduce the learning rate
2. Increase `num_mixtures` for more modeling capacity
3. Use gradient clipping (already implemented)

## Future Enhancements

1. **Attention Mechanisms**: Add attention to the MDN-RNN for better long-term dependencies
2. **Hierarchical Models**: Stack multiple levels of world models for hierarchical planning
3. **Curiosity-Driven Exploration**: Use prediction error as intrinsic reward
4. **Model-Based RL**: Implement full model-based algorithms like PETS or MBPO

## References

- [World Models](https://arxiv.org/abs/1803.10122) - Ha & Schmidhuber, 2018
- [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) - Hafner et al., 2018
- [Dream to Control](https://arxiv.org/abs/1912.01603) - Hafner et al., 2019

## License

This implementation is part of the JanusAI_V2 project and follows the same license terms.
