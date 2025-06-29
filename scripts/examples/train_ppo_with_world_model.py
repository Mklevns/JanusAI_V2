# File: janus/scripts/examples/train_ppo_with_world_model.py

''' Train a PPO agent using a world model that combines a VAE and MDN-RNN.
This script sets up the training environment, initializes the world model agent,
and runs the training loop with real and imagined rollouts.
'''

import logging
import random
import time
from pathlib import Path
from typing import Dict

import gymnasium as gym
import torch
import numpy as np

from janus.training.ppo.config import PPOConfig
from janus.training.ppo.trainer import PPOTrainer
from janus.training.ppo.buffer import RolloutBuffer
from janus.agents.world_model_agent import WorldModelAgent
from janus.agents.components.vae import VAEConfig
from janus.agents.components.mdn_rnn import MDNRNNConfig

logger = logging.getLogger(__name__)


class WorldModelPPOTrainer(PPOTrainer):
    def __init__(
        self,
        world_model_agent: WorldModelAgent,
        envs: list,
        config: PPOConfig,
        imagination_ratio: float = 0.5,
        imagination_horizon: int = 50,
        **kwargs
    ):
        super().__init__(
            agent=world_model_agent.controller,
            envs=envs,
            config=config,
            **kwargs
        )
        self.world_model_agent = world_model_agent
        self.imagination_ratio = imagination_ratio
        self.imagination_horizon = imagination_horizon
        self.global_step = 0
        self.imagination_buffer = RolloutBuffer(
            buffer_size=self.buffer.buffer_size,
            obs_shape=(world_model_agent.vae_config.latent_dim +
                       world_model_agent.mdn_config.hidden_dim,),
            action_shape=self.buffer.action_shape,
            device=self.device,
            n_envs=self.n_envs
        )

    def collect_rollouts(self, num_steps: int) -> Dict[str, float]:
        real_metrics = self._collect_real_rollouts(num_steps)
        if self.imagination_ratio > 0:
            imagination_steps = int(num_steps * self.imagination_ratio)
            imagination_metrics = self._collect_imagined_rollouts(imagination_steps)
            for key, value in imagination_metrics.items():
                real_metrics[f'imagination_{key}'] = value
        return real_metrics

    def _collect_real_rollouts(self, num_steps: int) -> Dict[str, float]:
        self.buffer.clear()
        collection_start = time.time()

        for _ in range(num_steps):
            obs_tensor = torch.tensor(self.collector.current_obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                actions, log_probs, components = self.world_model_agent.act(
                    obs_tensor,
                    deterministic=False,
                    return_components=True
                )
                controller_input = components['controller_input']
                values = self.agent.critic(controller_input).squeeze(-1)
            actions_np = actions.cpu().numpy()
            obs, rewards, dones, infos = self.collector.collect(
                lambda obs: (actions_np, log_probs.cpu().numpy()),
                self.executor
            )

            for i, done in enumerate(dones):
                if done:
                    obs[i], _ = self.envs[i].reset()

            self.buffer.add(
                obs=controller_input.cpu().numpy(),
                action=actions_np,
                reward=rewards,
                done=dones,
                value=values.cpu().numpy(),
                log_prob=log_probs.cpu().numpy()
            )
            self.global_step += self.n_envs

        stats = self.collector.get_statistics()
        stats['collection_time'] = time.time() - collection_start
        stats['steps_per_second'] = (num_steps * self.n_envs) / stats['collection_time']
        return stats

    def _collect_imagined_rollouts(self, num_steps: int) -> Dict[str, float]:
        self.imagination_buffer.clear()
        collection_start = time.time()

        current_obs = torch.tensor(self.collector.current_obs, dtype=torch.float32, device=self.device)
        self.world_model_agent.reset(batch_size=self.n_envs)
        trajectory = self.world_model_agent.imagine_trajectory(
            current_obs,
            horizon=min(self.imagination_horizon, num_steps // self.n_envs)
        )
        latents = trajectory['latents']
        actions = trajectory['actions']

        if actions is not None:
            for t in range(actions.shape[1]):
                hidden_dim = self.world_model_agent.mdn_config.hidden_dim
                hidden = torch.zeros(self.n_envs, hidden_dim, device=self.device)
                controller_input = torch.cat([latents[:, t], hidden], dim=-1)
                with torch.no_grad():
                    values = self.agent.critic(controller_input).squeeze(-1)
                    if self.continuous_actions:
                        log_probs = torch.zeros(self.n_envs, device=self.device)
                    else:
                        action_logits = self.agent.actor(controller_input)
                        action_probs = torch.softmax(action_logits, dim=-1)
                        log_probs = torch.log(
                            action_probs.gather(1, actions[:, t].long().unsqueeze(1))
                        ).squeeze(1)
                rewards = self._compute_imagined_rewards(
                    latents[:, t], latents[:, t + 1]
                )
                dones = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
                self.imagination_buffer.add(
                    obs=controller_input.cpu().numpy(),
                    action=actions[:, t].cpu().numpy(),
                    reward=rewards.cpu().numpy(),
                    done=dones.cpu().numpy(),
                    value=values.cpu().numpy(),
                    log_prob=log_probs.cpu().numpy()
                )

        imagination_time = time.time() - collection_start
        return {
            'steps': actions.shape[1] * self.n_envs if actions is not None else 0,
            'collection_time': imagination_time,
            'steps_per_second': (actions.shape[1] * self.n_envs) / imagination_time if actions is not None else 0
        }

    def _compute_imagined_rewards(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        latent_change = torch.norm(z_next - z_t, dim=-1)
        return torch.tanh(latent_change)

    def learn(self, current_clip_epsilon: float, current_entropy_coef: float) -> Dict[str, float]:
        real_data = self.buffer.get() if self.buffer.ptr > 0 else None
        imagined_data = self.imagination_buffer.get() if self.imagination_buffer.ptr > 0 else None
        all_metrics = {}
        if real_data is not None:
            real_metrics = self._learn_from_data(real_data, current_clip_epsilon, current_entropy_coef, prefix='real')
            all_metrics.update(real_metrics)
        if imagined_data is not None:
            imagined_metrics = self._learn_from_data(imagined_data, current_clip_epsilon, current_entropy_coef, prefix='imagined')
            all_metrics.update(imagined_metrics)
        return all_metrics

    def _learn_from_data(self, data: Dict[str, torch.Tensor], clip_epsilon: float, entropy_coef: float, prefix: str = '') -> Dict[str, float]:
        obs = torch.tensor(data['obs'], device=self.device, dtype=torch.float32)
        actions = torch.tensor(data['actions'], device=self.device)
        log_probs_old = torch.tensor(data['log_probs'], device=self.device)
        advantages = torch.tensor(data['advantages'], device=self.device)
        returns = torch.tensor(data['returns'], device=self.device)

        policy_losses, value_losses, entropies = [], [], []

        for _ in range(self.config.n_epochs):
            indices = np.random.permutation(len(obs))
            for start in range(0, len(obs), self.config.batch_size):
                end = start + self.config.batch_size
                batch_idx = indices[start:end]
                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_log_probs_old = log_probs_old[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                logits = self.agent.actor(batch_obs)
                dist = torch.distributions.Categorical(logits=logits) if not self.continuous_actions else self.agent.actor.get_dist(batch_obs)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.agent.critic(batch_obs).squeeze()
                value_loss = 0.5 * (batch_returns - values).pow(2).mean()

                loss = policy_loss + self.config.value_coef * value_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        return {
            f'{prefix}_policy_loss': np.mean(policy_losses),
            f'{prefix}_value_loss': np.mean(value_losses),
            f'{prefix}_entropy': np.mean(entropies)
        }



def train_world_model_ppo(env_name: str = "CarRacing-v2", total_timesteps: int = 1_000_000):
    logging.basicConfig(level=logging.INFO)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    envs = [gym.make(env_name) for _ in range(4)]
    checkpoint_dir = Path("checkpoints/world_model")
    obs, _ = envs[0].reset()
    obs_shape = obs.shape if obs.ndim == 3 else (1,) + obs.shape
    action_dim = envs[0].action_space.n if hasattr(envs[0].action_space, 'n') else envs[0].action_space.shape[0]
    continuous = not hasattr(envs[0].action_space, 'n')

    vae_config = VAEConfig(input_channels=obs_shape[0], input_height=64, input_width=64, latent_dim=32)
    mdn_config = MDNRNNConfig(latent_dim=32, action_dim=action_dim, hidden_dim=256)
    world_model_agent = WorldModelAgent(
        observation_shape=obs_shape,
        action_dim=action_dim,
        vae_config=vae_config,
        mdn_config=mdn_config,
        continuous_actions=continuous,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    if (checkpoint_dir / 'vae_final.pt').exists():
        world_model_agent.vae.load_state_dict(torch.load(checkpoint_dir / 'vae_final.pt'))
        logger.info("Loaded pre-trained VAE")
    if (checkpoint_dir / 'mdn_final.pt').exists():
        world_model_agent.mdn_rnn.load_state_dict(torch.load(checkpoint_dir / 'mdn_final.pt'))
        logger.info("Loaded pre-trained MDN-RNN")

    ppo_config = PPOConfig(
        learning_rate=3e-4,
        n_epochs=10,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        normalize_advantages=True,
        normalize_rewards=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    trainer = WorldModelPPOTrainer(
        world_model_agent=world_model_agent,
        envs=envs,
        config=ppo_config,
        imagination_ratio=0.5,
        imagination_horizon=50,
        experiment_name="world_model_ppo",
        use_tensorboard=True
    )

    logger.info("Starting World Model PPO training...")
    trainer.train(total_timesteps=total_timesteps, rollout_length=2048)
    world_model_agent.save_components(Path("checkpoints/world_model_ppo_final"))
    logger.info("Training completed!")
    for env in envs:
        env.close()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train_world_model_ppo(env_name="LunarLander-v2", total_timesteps=500_000)
