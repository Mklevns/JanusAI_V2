# janus/training/ppo/collector.py
"""Asynchronous Rollout Collector for PPO Training
This module implements an asynchronous rollout collector for PPO training,
allowing for efficient collection of environment interactions across multiple
parallel environments. It uses threading and a thread-safe design to ensure
robustness and performance, making it suitable for production-scale reinforcement
learning tasks.
"""


import numpy as np
import threading
import logging
import random
import torch
from typing import List, Tuple, Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

logger = logging.getLogger(__name__)


class AsyncRolloutCollector:
    """
    Asynchronous rollout collector for PPO training.

    Collects environment steps asynchronously across multiple environments. Supports
    safe retries, thread-safe statistics tracking, device-aware observations, and
    can be extended for subprocess-based parallelism. Supports returning `info`
    dictionaries for integration with reward handlers.
    """

    def __init__(
        self,
        envs: List,
        device: torch.device,
        num_workers: int = 4,
        use_subprocess: bool = False,
        seed: int = 42,
    ):
        """Initialize with reproducible seeding."""
        self.envs = envs
        self.device = device
        self.n_envs = len(envs)
        self.num_workers = min(num_workers, self.n_envs)
        self.use_subprocess = use_subprocess

        # Set seeds for each environment differently
        for i, env in enumerate(envs):
            env.seed(seed + i) # individual seeding
            # Also seed global random modules for general reproducibility outside envs
            # Though per-env seeding is more targeted for env behavior.
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # Initialize tracking with thread-safe statistics
        self.episode_rewards = [deque(maxlen=100) for _ in range(self.n_envs)]
        self.episode_lengths = [deque(maxlen=100) for _ in range(self.n_envs)]
        self.current_episode_rewards = np.zeros(self.n_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(self.n_envs, dtype=np.int32)
        self.last_valid_obs = [None] * self.n_envs
        self.lock = threading.RLock()  # Use RLock for re-entrant locking

        self.current_obs = self._reset_all_envs()
        logger.debug(
            "AsyncRolloutCollector initialized with %d environments", self.n_envs
        )

    def _reset_all_envs(self) -> np.ndarray:
        """Reset all environments and return initial observations."""
        obs_list = []
        for i, env in enumerate(self.envs):
            try:
                obs, _ = env.reset()
                obs_list.append(obs)
                self.last_valid_obs[i] = obs.copy()
            except Exception:
                logger.exception("Failed to reset environment %d", i)
                dummy_obs = np.zeros_like(self.envs[0].observation_space.sample())
                obs_list.append(dummy_obs)
                self.last_valid_obs[i] = dummy_obs
        return np.array(obs_list)

    def _step_env_safe(
        self, env_idx: int, action: np.ndarray, max_retries: int = 3
    ) -> Tuple[int, np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment with comprehensive error handling."""
        if not isinstance(action, np.ndarray): # Basic type check
            try:
                action = np.array(action) # Attempt conversion if not numpy array
            except Exception as e:
                logger.error(f"Failed to convert action to numpy array for env {env_idx}: {e}")
                # Attempt to use a default/dummy action or raise error immediately
                # For now, let's try to use a zero action if possible, or re-raise
                if hasattr(self.envs[env_idx].action_space, 'sample'):
                    action = np.zeros_like(self.envs[env_idx].action_space.sample())
                else: # Cannot create a sensible default
                    raise ValueError(f"Action for env {env_idx} is not a np.ndarray and cannot be converted or defaulted.")


        if action.shape != self.envs[env_idx].action_space.shape:
            # Before raising, check if action_space is discrete and action is a scalar
            is_discrete = hasattr(self.envs[env_idx].action_space, 'n')
            is_scalar_action_for_discrete = is_discrete and action.ndim == 0

            if not (is_discrete and action.ndim == 1 and action.shape[0] == 1) and \
               not is_scalar_action_for_discrete : # Allow single int in array for discrete
                # If action is scalar for discrete, it's often passed as (e.g.) np.array(2) not np.array([2])
                # Some envs expect int, others np.array([int]). The check above is a bit lenient.
                # A common pattern is discrete actions are int, continuous are float arrays.
                # The provided check `action.shape != self.envs[env_idx].action_space.shape`
                # might be too strict if `action_space.shape` is `()` for discrete but action is `(1,)`.
                # For now, let's assume the original check is intended.
                 raise ValueError(
                    f"Action shape mismatch for env {env_idx}: "
                    f"expected {self.envs[env_idx].action_space.shape}, got {action.shape}"
                )

        for attempt in range(max_retries):
            try:
                step_output = self.envs[env_idx].step(action)

                # Handle both Gym and Gymnasium APIs
                if len(step_output) == 4:
                    next_obs, reward, done, info = step_output
                    truncated = False #gym.Env
                elif len(step_output) == 5:
                    next_obs, reward, done, truncated, info = step_output #gymnasium.Env
                else:
                    raise ValueError(
                        f"Unexpected step output length for env {env_idx}: {len(step_output)}"
                    )

                # Validate outputs
                if not isinstance(reward, (int, float, np.number)):
                    logger.warning(f"Env {env_idx}: Reward is not numeric ({type(reward)}), attempting to cast.")
                    try:
                        reward = float(reward)
                    except (ValueError, TypeError) as e:
                        logger.error(f"Env {env_idx}: Failed to cast reward to float: {e}. Using 0.0.")
                        reward = 0.0
                        # raise TypeError(f"Reward must be numeric, got {type(reward)}")
                if not isinstance(done, (bool, np.bool_)):
                    logger.warning(f"Env {env_idx}: Done is not boolean ({type(done)}), attempting to cast.")
                    try:
                        done = bool(done)
                    except (ValueError, TypeError) as e:
                        logger.error(f"Env {env_idx}: Failed to cast done to bool: {e}. Using True (terminating).")
                        done = True # Terminate on unknown state
                        # raise TypeError(f"Done must be boolean, got {type(done)}")
                if not isinstance(truncated, (bool, np.bool_)): # Added check for truncated
                    logger.warning(f"Env {env_idx}: Truncated is not boolean ({type(truncated)}), attempting to cast.")
                    try:
                        truncated = bool(truncated)
                    except (ValueError, TypeError) as e:
                        logger.error(f"Env {env_idx}: Failed to cast truncated to bool: {e}. Using False.")
                        truncated = False


                self.last_valid_obs[env_idx] = next_obs.copy()
                return env_idx, next_obs, float(reward), bool(done), bool(truncated), info

            except Exception as e:
                logger.error(
                    "Env %d step failed (attempt %d/%d): %s",
                    env_idx, attempt + 1, max_retries, str(e),
                    exc_info=True # adding exc_info for more details
                )
                if attempt == max_retries - 1:
                    logger.warning(
                        "Max retries reached for env %d, attempting reset", env_idx
                    )
                    return self._reset_env_on_failure(env_idx)

        # This part should ideally not be reached if max_retries > 0 because
        # the loop either returns successfully or calls _reset_env_on_failure.
        # However, to satisfy linters/type checkers that a return is guaranteed:
        return self._reset_env_on_failure(env_idx)

    def _reset_env_on_failure(
        self, env_idx: int
    ) -> Tuple[int, np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Reset a failed environment and return a terminal transition with dummy info."""
        try:
            reset_output = self.envs[env_idx].reset()
            if isinstance(reset_output, tuple):
                obs, info = reset_output
            else:
                obs = reset_output
                info = {}
            self.last_valid_obs[env_idx] = obs.copy()
            return env_idx, obs, 0.0, True, False, info
        except Exception:
            logger.exception(
                "CRITICAL: Failed to reset env %d after multiple errors.", env_idx
            )
            return env_idx, self.last_valid_obs[env_idx], 0.0, True, False, {}

    def _process_step_results(
        self, step_results: List[Tuple]
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]
    ]:
        """Process list of step results and update internal state."""
        step_results.sort(key=lambda x: x[0])
        next_obs_list, rewards, dones, truncateds, infos = [], [], [], [], []

        for _, next_obs, reward, done, truncated, info in step_results:
            next_obs_list.append(next_obs)
            rewards.append(reward)
            dones.append(done)
            truncateds.append(truncated)
            infos.append(info)

        with self.lock:
            rewards_np = np.array(rewards, dtype=np.float32)
            self.current_episode_rewards += rewards_np
            self.current_episode_lengths += 1
            for i in range(self.n_envs):
                if dones[i] or truncateds[i]:
                    self.episode_rewards[i].append(self.current_episode_rewards[i])
                    self.episode_lengths[i].append(self.current_episode_lengths[i])
                    self.current_episode_rewards[i] = 0.0
                    self.current_episode_lengths[i] = 0
                    try:
                        obs, _ = self.envs[i].reset()
                        next_obs_list[i] = obs
                        self.last_valid_obs[i] = obs.copy()
                    except Exception:
                        logger.exception("Failed to reset env %d after episode", i)
                        next_obs_list[i] = self.last_valid_obs[i]

        self.current_obs = np.array(next_obs_list)
        return (
            torch.tensor(self.current_obs).to(self.device),
            torch.tensor(rewards_np).to(self.device),
            torch.tensor(dones, dtype=torch.bool).to(self.device),
            torch.tensor(truncateds, dtype=torch.bool).to(self.device),
            infos,
        )

    def collect(
        self,
        get_actions_fn: Callable[[torch.Tensor], Tuple[np.ndarray, Any]],
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> Tuple[
        np.ndarray,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Any,
        List[Dict[str, Any]],
    ]:
        """Collect one rollout step across all environments."""
        obs_tensor = torch.from_numpy(self.current_obs).to(self.device)
        actions_np, log_probs = get_actions_fn(obs_tensor)
        original_obs = self.current_obs.copy()

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

        next_obs, rewards, dones, truncateds, infos = self._process_step_results(
            results
        )

        return original_obs, next_obs, rewards, dones, truncateds, log_probs, infos

    def get_statistics(self) -> dict:
        """Return aggregated statistics of recent episodes."""
        with self.lock:
            valid_rewards = [r for r in self.episode_rewards if len(r) > 0]
            valid_lengths = [l for l in self.episode_lengths if len(l) > 0]

            if not valid_rewards:
                return {
                    "mean_episode_reward": 0.0,
                    "mean_episode_length": 0.0,
                    "min_episode_reward": 0.0,
                    "max_episode_reward": 0.0,
                    "std_episode_reward": 0.0,
                    "total_episodes": 0,
                }

            all_rewards = [r for q in valid_rewards for r in q]
            all_lengths = [l for q in valid_lengths for l in q]

            return {
                "mean_episode_reward": np.mean(all_rewards),
                "mean_episode_length": np.mean(all_lengths) if all_lengths else 0.0,
                "min_episode_reward": np.min(all_rewards),
                "max_episode_reward": np.max(all_rewards),
                "std_episode_reward": np.std(all_rewards),
                "total_episodes": len(all_rewards),
            }
