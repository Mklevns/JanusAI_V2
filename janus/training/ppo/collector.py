# janus/training/ppo/collector.py
"""
Rollout Collector for Proximal Policy Optimization (PPO) training.

This module implements an asynchronous rollout collector that can handle multiple 
environments, supports safe stepping with retries, and collects statistics on 
episode rewards and lengths. It is designed to be thread-safe and can be used 
with or without subprocesses for parallel execution.
"""

import numpy as np
import threading
import logging
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed  # Fixed: Added missing import
from collections import deque

logger = logging.getLogger(__name__)


class AsyncRolloutCollector:
    """
    Asynchronous rollout collector for parallel environment interaction.
    
    Features:
        - Thread-safe episode statistics tracking
        - Robust environment stepping with retries
        - Support for parallel execution with ThreadPoolExecutor
        - Automatic environment reset on episode termination
    """
    
    def __init__(
        self, 
        envs: List, 
        device,
        num_workers: int = 4, 
        use_subprocess: bool = False
    ):
        """
        Initialize the async rollout collector.
        
        Args:
            envs: List of environment instances
            device: PyTorch device for tensor operations
            num_workers: Maximum number of parallel workers
            use_subprocess: Whether to use subprocess isolation (future feature)
        """
        self.envs = envs
        self.device = device
        self.n_envs = len(envs)
        self.num_workers = min(num_workers, self.n_envs)
        self.use_subprocess = use_subprocess
        
        # Episode tracking with thread safety
        self.episode_rewards = [deque(maxlen=100) for _ in range(self.n_envs)]
        self.episode_lengths = [deque(maxlen=100) for _ in range(self.n_envs)]
        self.current_episode_rewards = np.zeros(self.n_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(self.n_envs, dtype=np.int32)
        self.last_valid_obs = [None] * self.n_envs
        self.lock = threading.Lock()
        
        # Initialize all environments
        self.current_obs = self._reset_all_envs()
        
        logger.debug(f"AsyncRolloutCollector initialized with {self.n_envs} environments")
        
    def _reset_all_envs(self) -> np.ndarray:
        """
        Reset all environments and store initial observations.
        
        Returns:
            Array of initial observations from all environments
        """
        obs_list = []
        for i, env in enumerate(self.envs):
            try:
                obs, _ = env.reset()
                obs_list.append(obs)
                self.last_valid_obs[i] = obs.copy()
            except Exception as e:
                logger.error(f"Failed to reset environment {i}: {e}")
                # Use zeros as fallback
                dummy_obs = np.zeros_like(self.envs[0].observation_space.sample())
                obs_list.append(dummy_obs)
                self.last_valid_obs[i] = dummy_obs
                
        return np.array(obs_list)
        
    def _step_env_safe(
        self, 
        env_idx: int, 
        action: np.ndarray, 
        max_retries: int = 3
    ) -> Tuple[np.ndarray, float, bool, bool]:
        """
        Safely step a single environment with retry logic.
        
        Args:
            env_idx: Index of the environment to step
            action: Action to take in the environment
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (next_observation, reward, done, truncated)
        """
        for attempt in range(max_retries):
            try:
                # Step the environment
                next_obs, reward, done, truncated, info = self.envs[env_idx].step(action)
                self.last_valid_obs[env_idx] = next_obs.copy()
                return next_obs, reward, done, truncated
                
            except Exception as e:
                logger.error(f"Env {env_idx} step failed (attempt {attempt+1}/{max_retries}): {e}")
                
                if attempt == max_retries - 1:
                    # Final attempt failed, try to reset the environment
                    logger.warning(f"Max retries reached for env {env_idx}, attempting reset")
                    try:
                        obs, _ = self.envs[env_idx].reset()
                        self.last_valid_obs[env_idx] = obs.copy()
                        return obs, 0.0, True, False
                    except Exception as reset_error:
                        logger.error(f"Failed to reset env {env_idx}: {reset_error}")
                        # Return last valid observation as final fallback
                        return self.last_valid_obs[env_idx], 0.0, True, False
                        
    def collect_steps(
        self, 
        actions: np.ndarray, 
        executor: Optional[ThreadPoolExecutor] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect a single step from all environments.
        
        Args:
            actions: Array of actions for each environment
            executor: Optional ThreadPoolExecutor for parallel execution
            
        Returns:
            Tuple of (next_observations, rewards, dones, truncateds)
        """
        if executor is None:
            # Sequential execution
            results = []
            for i, action in enumerate(actions):
                result = self._step_env_safe(i, action)
                results.append((i, *result))
        else:
            # Parallel execution
            futures = {
                executor.submit(self._step_env_safe, i, action): i 
                for i, action in enumerate(actions)
            }
            results = []
            for future in as_completed(futures):
                env_idx = futures[future]
                try:
                    result = future.result()
                    results.append((env_idx, *result))
                except Exception as e:
                    logger.error(f"Future failed for env {env_idx}: {e}")
                    # Use fallback values
                    results.append((env_idx, self.last_valid_obs[env_idx], 0.0, True, False))
                    
        # Sort results by environment index to maintain order
        results.sort(key=lambda x: x[0])
        
        # Extract components
        next_obs_list, rewards, dones, truncateds = [], [], [], []
        for _, next_obs, reward, done, truncated in results:
            next_obs_list.append(next_obs)
            rewards.append(reward)
            dones.append(done)
            truncateds.append(truncated)
            
        # Update episode statistics (thread-safe)
        with self.lock:
            self.current_episode_rewards += np.array(rewards)
            self.current_episode_lengths += 1
            
            # Handle episode termination
            for i in range(self.n_envs):
                if dones[i] or truncateds[i]:
                    # Record completed episode
                    self.episode_rewards[i].append(self.current_episode_rewards[i])
                    self.episode_lengths[i].append(self.current_episode_lengths[i])
                    
                    # Reset episode tracking
                    self.current_episode_rewards[i] = 0.0
                    self.current_episode_lengths[i] = 0
                    
                    # Reset environment
                    try:
                        obs, _ = self.envs[i].reset()
                        next_obs_list[i] = obs
                        self.last_valid_obs[i] = obs.copy()
                    except Exception as e:
                        logger.error(f"Failed to reset env {i} after episode: {e}")
                        # Keep the last observation
                        next_obs_list[i] = self.last_valid_obs[i]
                        
        # Update current observations
        self.current_obs = np.array(next_obs_list)
        
        return (
            self.current_obs,
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(truncateds, dtype=bool)
        )
        
    def get_statistics(self) -> dict:
        """
        Get episode statistics in a thread-safe manner.
        
        Returns:
            Dictionary containing:
                - mean_episode_reward: Average episode reward across all environments
                - mean_episode_length: Average episode length
                - min_episode_reward: Minimum episode reward
                - max_episode_reward: Maximum episode reward
        """
        with self.lock:
            # Filter out empty deques
            valid_rewards = [r for r in self.episode_rewards if len(r) > 0]
            valid_lengths = [l for l in self.episode_lengths if len(l) > 0]
            
            if not valid_rewards:
                return {
                    "mean_episode_reward": 0.0,
                    "mean_episode_length": 0.0,
                    "min_episode_reward": 0.0,
                    "max_episode_reward": 0.0,
                }
                
            # Calculate statistics
            all_rewards = []
            for reward_deque in valid_rewards:
                all_rewards.extend(list(reward_deque))
                
            all_lengths = []
            for length_deque in valid_lengths:
                all_lengths.extend(list(length_deque))
                
            return {
                "mean_episode_reward": np.mean(all_rewards),
                "mean_episode_length": np.mean(all_lengths) if all_lengths else 0.0,
                "min_episode_reward": np.min(all_rewards),
                "max_episode_reward": np.max(all_rewards),
                "std_episode_reward": np.std(all_rewards),
                "total_episodes": len(all_rewards),
            }