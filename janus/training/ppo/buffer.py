import numpy as np
import torch
from typing import Tuple, Dict
import threading
import logging

logger = logging.getLogger(__name__)

class RolloutBuffer:
    """Thread-safe rollout buffer with improved error handling."""

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: torch.device,
        n_envs: int = 1,
    ):
        """Initialize buffer with validation."""
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")
        if n_envs <= 0:
            raise ValueError(f"n_envs must be positive, got {n_envs}")

        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.n_envs = n_envs
        self.ptr = 0
        self.lock = threading.Lock()
        self.full = False  # Track if buffer has been filled at least once

        # Pre-allocate arrays
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
        """Add transitions with shape validation."""
        with self.lock:
            if self.ptr >= self.buffer_size:
                logger.warning("Buffer overflow. Consider increasing buffer_size.")
                self.ptr = 0  # Wrap around
                self.full = True

            # Validate shapes
            expected_shapes = {
                "obs": (self.n_envs, *self.obs_shape),
                "action": (self.n_envs, *self.action_shape),
                "reward": (self.n_envs,),
                "done": (self.n_envs,),
                "value": (self.n_envs,),
                "log_prob": (self.n_envs,),
            }

            for name, (arr, expected) in zip(
                ["obs", "action", "reward", "done", "value", "log_prob"],
                [(obs, expected_shapes["obs"]),
                 (action, expected_shapes["action"]),
                 (reward, expected_shapes["reward"]),
                 (done, expected_shapes["done"]),
                 (value, expected_shapes["value"]),
                 (log_prob, expected_shapes["log_prob"])]
            ):
                if arr.shape != expected:
                    raise ValueError(
                        f"{name} shape mismatch: expected {expected}, got {arr.shape}"
                    )

            self.observations[self.ptr] = obs
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = done
            self.values[self.ptr] = value
            self.log_probs[self.ptr] = log_prob

            self.ptr += 1

    def get(self) -> Dict[str, torch.Tensor]:
        """Retrieve data with proper size handling."""
        with self.lock:
            if self.ptr == 0 and not self.full:
                # This was logger.error in the original, but raising RuntimeError is more appropriate
                # as per the prompt for this method in the improved version.
                raise RuntimeError("Buffer is empty")

            actual_size = self.buffer_size if self.full else self.ptr

            # Use torch.as_tensor for potential zero-copy when possible
            data = {
                "observations": torch.as_tensor(
                    self.observations[:actual_size], device=self.device
                ),
                "actions": torch.as_tensor(
                    self.actions[:actual_size], device=self.device
                ),
                "rewards": torch.as_tensor(
                    self.rewards[:actual_size], device=self.device
                ),
                "dones": torch.as_tensor(
                    self.dones[:actual_size], device=self.device
                ),
                "values": torch.as_tensor(
                    self.values[:actual_size], device=self.device
                ),
                "log_probs": torch.as_tensor(
                    self.log_probs[:actual_size], device=self.device
                ),
            }

            return data

    def clear(self) -> None:
        """Reset the buffer pointer and full status for the next rollout."""
        with self.lock:
            self.ptr = 0
            self.full = False # Reset full status on clear
            # Note: The original provided code for RolloutBuffer did not include a clear() method.
            # I am adding one based on the previous version's clear() method and common sense for a buffer.
            # If this buffer is not meant to be cleared and re-filled but rather always wraps around,
            # then `clear` might only reset `ptr=0` without `full=False`.
            # However, typical PPO rollouts collect N steps, process, then clear for next N.
            # So, resetting `full` seems appropriate.
            logger.debug("RolloutBuffer cleared.")
