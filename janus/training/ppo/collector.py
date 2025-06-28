# janus/training/ppo/collector.py
''' Asynchronous Rollout Collector for PPO Training
This module implements an asynchronous rollout collector for PPO training,
allowing for efficient collection of environment interactions across multiple
parallel environments. It uses threading and a thread-safe design to ensure
robustness and performance, making it suitable for production-scale reinforcement
learning tasks.
'''


import logging
import random
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class AsyncRolloutCollector:
    """
    Asynchronous rollout collector for PPO training.

    This refactored version improves modularity by breaking down the monolithic
    `collect_steps` method into smaller, single-responsibility functions. This
    aligns with best practices for creating maintainable and debuggable ML
    systems.

    Key Improvements:
        - **Separation of Concerns**: Logic is split into distinct methods for
          dispatching jobs, processing results, and handling episode ends.
        - **Improved Readability**: Smaller methods with clear names make the
          data flow easier to follow.
        - **Enhanced Modularity**: Core logic, like handling episode termination,
          is encapsulated in its own function, making it easier to modify
          or extend.
    """

    def __init__(
        self,
        envs: List[Any],
        device: torch.device,
        num_workers: int = 4,
        use_subprocess: bool = False,
        seed: int = 42,
    ):
        """
        Initialize the AsyncRolloutCollector.

        Args:
            envs (List): List of environment instances following Gym API.
            device: PyTorch device to which observations and tensors should be moved.
            num_workers (int): Number of parallel threads to use.
            use_subprocess (bool): Placeholder for subprocess-based env execution.
            seed (int): Seed for reproducibility across random generators.
        """
        self.envs = envs
        self.device = device
        self.n_envs = len(envs)
        self.num_workers = min(num_workers, self.n_envs)
        self.use_subprocess = use_subprocess
        self.seed = seed

        # Seeding for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Thread-safe episode statistics tracking
        self.episode_rewards = [deque(maxlen=100) for _ in range(self.n_envs)]
        self.episode_lengths = [deque(maxlen=100) for _ in range(self.n_envs)]
        self.current_episode_rewards = np.zeros(self.n_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(self.n_envs, dtype=np.int32)
        self.last_valid_obs = [None] * self.n_envs
        self.lock = threading.Lock()

        # Initialize environments and get initial observations
        self.current_obs = self._reset_all_envs()
        logger.debug("AsyncRolloutCollector initialized with %d environments", self.n_envs)

    def _reset_all_envs(self) -> np.ndarray:
        """
        Reset all environments safely and return initial observations.

        Returns:
            np.ndarray: Array of initial observations.
        """
        obs_list = []
        for i, env in enumerate(self.envs):
            try:
                obs, _ = env.reset(seed=self.seed + i)
                obs_list.append(obs)
                self.last_valid_obs[i] = obs.copy()
            except Exception:
                logger.exception("Failed to reset environment %d", i)
                # Create a dummy observation based on the first env's space
                dummy_obs = np.zeros_like(self.envs[0].observation_space.sample())
                obs_list.append(dummy_obs)
                self.last_valid_obs[i] = dummy_obs
        return np.array(obs_list)

    def _step_env_safe(
        self, env_idx: int, action: np.ndarray, max_retries: int = 3
    ) -> Tuple[int, np.ndarray, float, bool, bool]:
        """
        Safely steps a single environment with retries and robust error handling.
        This function is designed to be called by the ThreadPoolExecutor.
        """
        for attempt in range(max_retries):
            try:
                next_obs, reward, done, truncated, _ = self.envs[env_idx].step(action)
                self.last_valid_obs[env_idx] = next_obs.copy()
                return env_idx, next_obs, reward, done, truncated
            except Exception:
                logger.exception(
                    "Env %d step failed (attempt %d/%d)", env_idx, attempt + 1, max_retries
                )
                if attempt == max_retries - 1:
                    logger.warning(
                        "Max retries reached for env %d, attempting reset.", env_idx
                    )
                    return self._reset_env_on_failure(env_idx)
        # This line should ideally be unreachable
        return self._reset_env_on_failure(env_idx)

    def _reset_env_on_failure(
        self, env_idx: int
    ) -> Tuple[int, np.ndarray, float, bool, bool]:
        """Resets a single environment after a critical failure and returns a terminal state."""
        try:
            obs, _ = self.envs[env_idx].reset()
            self.last_valid_obs[env_idx] = obs.copy()
            return env_idx, obs, 0.0, True, False  # Mark as 'done' to reset agent state
        except Exception:
            logger.exception("CRITICAL: Failed to reset env %d after multiple errors.", env_idx)
            # Return last known valid observation to prevent system crash
            return env_idx, self.last_valid_obs[env_idx], 0.0, True, False

    def _handle_episode_termination(
        self, env_idx: int, next_obs_list: List[np.ndarray]
    ):
        """
        Handles the end of an episode for a single environment.
        This includes logging stats, resetting counters, and resetting the environment.
        This function is called within a thread-safe context.
        """
        self.episode_rewards[env_idx].append(self.current_episode_rewards[env_idx])
        self.episode_lengths[env_idx].append(self.current_episode_lengths[env_idx])
        self.current_episode_rewards[env_idx] = 0.0
        self.current_episode_lengths[env_idx] = 0
        try:
            obs, _ = self.envs[env_idx].reset()
            next_obs_list[env_idx] = obs
            self.last_valid_obs[env_idx] = obs.copy()
        except Exception:
            logger.exception("Failed to reset env %d after episode completion.", env_idx)
            # Use last valid observation on failure
            next_obs_list[env_idx] = self.last_valid_obs[env_idx]

    def _process_step_results(
        self, step_results: List[Tuple]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes the raw results from the environment steps.

        This method sorts results, updates episode statistics, handles environment
        resets, and updates the collector's internal state. It encapsulates the
        logic that was previously crowded inside `collect_steps`.
        """
        # Sort results by environment index to ensure correct order
        step_results.sort(key=lambda x: x[0])

        # Unpack results into separate lists
        next_obs_list, rewards, dones, truncateds = [], [], [], []
        for _, next_obs, reward, done, truncated in step_results:
            next_obs_list.append(next_obs)
            rewards.append(reward)
            dones.append(done)
            truncateds.append(truncated)

        # Update episode statistics and handle resets in a thread-safe manner
        with self.lock:
            rewards_np = np.array(rewards, dtype=np.float32)
            self.current_episode_rewards += rewards_np
            self.current_episode_lengths += 1
            for i in range(self.n_envs):
                if dones[i] or truncateds[i]:
                    self._handle_episode_termination(i, next_obs_list)

        # Update the main observation buffer
        self.current_obs = np.array(next_obs_list, dtype=np.float32)

        return self.current_obs, rewards_np, np.array(dones), np.array(truncateds)

    def collect(
        self,
        get_actions_fn: Callable[[torch.Tensor], Tuple[np.ndarray, Any]],
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        """
        Performs a single step of asynchronous rollout collection.

        Args:
            get_actions_fn: A function that takes a batch of observations (as a
                            torch.Tensor) and returns a tuple of actions (np.ndarray)
                            and their log probabilities (any).
            executor: A ThreadPoolExecutor for parallel execution.

        Returns:
            A tuple containing:
            - The observations before the step.
            - The resulting next observations.
            - The rewards received.
            - The done flags.
            - The truncated flags.
            - The log probabilities for the actions taken.
        """
        # --- Action Selection Phase ---
        obs_tensor = torch.from_numpy(self.current_obs).to(self.device)
        actions_np, log_probs = get_actions_fn(obs_tensor)

        # Keep a copy of observations before the step
        original_obs = self.current_obs.copy()

        # --- Environment Stepping Phase ---
        if executor:
            futures = [
                executor.submit(self._step_env_safe, i, actions_np[i])
                for i in range(self.n_envs)
            ]
            results = [future.result() for future in as_completed(futures)]
        else:
            results = [
                self._step_env_safe(i, actions_np[i]) for i in range(self.n_envs)
            ]

        # --- Result Processing Phase ---
        next_obs, rewards, dones, truncateds = self._process_step_results(results)

        return original_obs, next_obs, rewards, dones, truncateds, log_probs

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieves and calculates episode statistics in a thread-safe manner.
        """
        with self.lock:
            # Flatten the list of deques
            all_rewards = [r for deque_ in self.episode_rewards for r in deque_]
            all_lengths = [l for deque_ in self.episode_lengths for l in deque_]

            if not all_rewards:
                return {
                    "mean_episode_reward": 0.0,
                    "std_episode_reward": 0.0,
                    "mean_episode_length": 0.0,
                    "min_episode_reward": 0.0,
                    "max_episode_reward": 0.0,
                    "total_episodes": 0,
                }

            return {
                "mean_episode_reward": np.mean(all_rewards),
                "std_episode_reward": np.std(all_rewards),
                "mean_episode_length": np.mean(all_lengths),
                "min_episode_reward": np.min(all_rewards),
                "max_episode_reward": np.max(all_rewards),
                "total_episodes": len(all_rewards),
            }

    def reset(self) -> np.ndarray:
        """
        Resets all environments and returns the initial observations.
        """
        self.current_obs = self._reset_all_envs()
        return self.current_obs