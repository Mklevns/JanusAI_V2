# janus/training/ppo/buffer.py

''' Rollout buffer for storing transitions in PPO training.
This module implements a thread-safe buffer for storing observations, actions,
rewards, dones, values, and log probabilities during PPO training.
'''


import numpy as np
import torch
from typing import Tuple, Dict
import threading
import logging

logger = logging.getLogger(__name__)


class RolloutBuffer:
    """
    A buffer for storing rollouts from multiple environments for PPO training.

    Attributes:
        buffer_size (int): Number of time steps to store.
        obs_shape (Tuple[int, ...]): Shape of observations.
        action_shape (Tuple[int, ...]): Shape of actions.
        device (torch.device): Torch device to place tensors on.
        n_envs (int): Number of parallel environments.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: torch.device,
        n_envs: int = 1,
    ):
        """Initialize the rollout buffer with preallocated arrays."""
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.n_envs = n_envs
        self.ptr = 0
        self.lock = threading.Lock()

        self.observations = np.zeros((buffer_size, n_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_envs, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
    ) -> None:
        """
        Add a new set of transitions to the buffer.

        Args:
            obs: Observations.
            action: Actions taken.
            reward: Rewards received.
            done: Done flags.
            value: Value function estimates.
            log_prob: Log probabilities of actions.
        """
        with self.lock:
            if self.ptr >= self.buffer_size:
                logger.warning("Buffer overflow detected.")
                return

            self.observations[self.ptr] = obs
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = done
            self.values[self.ptr] = value
            self.log_probs[self.ptr] = log_prob

            self.ptr += 1

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve collected data as a dictionary of torch tensors.

        Returns:
            A dictionary with keys: observations, actions, rewards, dones, values, log_probs.
        """
        with self.lock:
            assert self.ptr > 0, "Buffer is empty"
            actual_size = self.ptr

            data = {
                "observations": torch.from_numpy(self.observations[:actual_size].copy()).to(self.device),
                "actions": torch.from_numpy(self.actions[:actual_size].copy()).to(self.device),
                "rewards": torch.from_numpy(self.rewards[:actual_size].copy()).to(self.device),
                "dones": torch.from_numpy(self.dones[:actual_size].copy()).to(self.device),
                "values": torch.from_numpy(self.values[:actual_size].copy()).to(self.device),
                "log_probs": torch.from_numpy(self.log_probs[:actual_size].copy()).to(self.device),
            }

            return data

    def clear(self) -> None:
        """Reset the buffer pointer for the next rollout."""
        with self.lock:
            self.ptr = 0
