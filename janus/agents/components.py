"""Custom reward components for specific Janus AI tasks."""

from typing import Any, Dict, Optional, List
import numpy as np
from base import BaseRewardComponent  # Changed from relative to absolute import


class SymbolicRegressionReward(BaseRewardComponent):
    """Reward component for symbolic regression tasks.

    Rewards the agent for generating expressions with high accuracy and simplicity.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config: Dictionary containing configuration parameters:
                - accuracy_weight (float): Weight for accuracy reward.
                - parsimony_weight (float): Weight for expression length penalty.
                - target_mse (float, optional): MSE threshold to activate parsimony bonus.
        """
        super().__init__(config)
        self.accuracy_weight = config["accuracy_weight"]
        self.parsimony_weight = config["parsimony_weight"]
        self.target_mse = config.get("target_mse", 0.01)

    def compute(
        self,
        _observation: np.ndarray,
        _action: np.ndarray,
        _next_observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute reward based on MSE and expression complexity.

        Args:
            _observation: Unused.
            _action: Unused.
            _next_observation: Unused.
            info: Dictionary with 'mse' and 'expression_length'.

        Returns:
            A float reward score.
        """
        if info is None:
            return 0.0

        mse = info.get("mse", float("inf"))
        expr_length = info.get("expression_length", 0)

        accuracy_reward = self.accuracy_weight / (1.0 + mse)
        parsimony_bonus = (
            self.parsimony_weight / (1.0 + expr_length)
            if mse <= self.target_mse
            else 0.0
        )

        return accuracy_reward + parsimony_bonus


class CommunicationEfficiencyReward(BaseRewardComponent):
    """Reward for minimizing communication overhead and encouraging task success."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config: Dictionary containing configuration parameters:
                - message_penalty (float): Penalty per message over bandwidth limit.
                - success_bonus (float): Bonus if cooperative task succeeds.
                - bandwidth_limit (int, optional): Max free-message bandwidth.
        """
        super().__init__(config)
        self.message_penalty = config["message_penalty"]
        self.success_bonus = config["success_bonus"]
        self.bandwidth_limit = config.get("bandwidth_limit", 10)

    def compute(
        self,
        _observation: np.ndarray,
        _action: np.ndarray,
        _next_observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute reward based on communication usage and task success.

        Args:
            _observation: Unused.
            _action: Unused.
            _next_observation: Unused.
            info: Dictionary with 'messages_sent' and 'cooperative_task_success'.

        Returns:
            A float reward score.
        """
        if info is None:
            return 0.0

        messages_sent = info.get("messages_sent", 0)
        task_success = info.get("cooperative_task_success", False)

        comm_penalty = -self.message_penalty * max(
            0, messages_sent - self.bandwidth_limit
        )
        success_reward = self.success_bonus if task_success else 0.0

        return comm_penalty + success_reward


class AdaptiveDifficultyReward(BaseRewardComponent):
    """Adjusts difficulty based on recent performance to keep agent learning efficiently."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config: Dictionary containing configuration parameters:
                - base_reward (float): Base reward when task is completed.
                - difficulty_scale (float): Scaling factor for difficulty.
                - success_threshold (float, optional): Desired success rate.
                - window_size (int, optional): Number of past steps to track.
        """
        super().__init__(config)
        self.base_reward = config["base_reward"]
        self.difficulty_scale = config["difficulty_scale"]
        self.success_threshold = config.get("success_threshold", 0.8)
        self.recent_successes: List[float] = []
        self.window_size = config.get("window_size", 100)
        self.current_difficulty = 1.0

    def compute(
        self,
        _observation: np.ndarray,
        _action: np.ndarray,
        _next_observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute reward scaled by adaptive difficulty.

        Args:
            _observation: Unused.
            _action: Unused.
            _next_observation: Unused.
            info: Dictionary with 'task_completed' (bool).

        Returns:
            A float reward score scaled by current difficulty.
        """
        if info is None:
            return 0.0

        task_success = info.get("task_completed", False)
        self._update_performance(task_success)
        self._adjust_difficulty()

        return self.base_reward * self.current_difficulty if task_success else 0.0

    def _update_performance(self, success: bool) -> None:
        """Record task outcome into rolling history.

        Args:
            success: Whether the agent completed the task.
        """
        self.recent_successes.append(float(success))
        if len(self.recent_successes) > self.window_size:
            self.recent_successes.pop(0)

    def _adjust_difficulty(self) -> None:
        """Update current difficulty based on recent success rate."""
        if len(self.recent_successes) < self.window_size // 2:
            return

        success_rate = np.mean(self.recent_successes)
        if success_rate > self.success_threshold:
            self.current_difficulty *= 1.1
        elif success_rate < self.success_threshold * 0.5:
            self.current_difficulty *= 0.9

        self.current_difficulty = np.clip(self.current_difficulty, 0.1, 10.0)

    def reset(self) -> None:
        """Reset adaptive history and difficulty to default state."""
        self.recent_successes = []
        self.current_difficulty = 1.0
