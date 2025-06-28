"""Custom reward components for specific Janus AI tasks."""

from typing import Any, Dict, Optional
import numpy as np
from .base import BaseRewardComponent


class SymbolicRegressionReward(BaseRewardComponent):
    """Reward component for symbolic regression tasks."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.accuracy_weight = config['accuracy_weight']
        self.parsimony_weight = config['parsimony_weight']
        self.target_mse = config.get('target_mse', 0.01)

    def compute(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        if info is None:
            return 0.0

        mse = info.get('mse', float('inf'))
        expr_length = info.get('expression_length', 0)

        accuracy_reward = self.accuracy_weight / (1.0 + mse)
        parsimony_bonus = (
            self.parsimony_weight / (1.0 + expr_length)
            if mse <= self.target_mse else 0.0
        )

        return accuracy_reward + parsimony_bonus


class CommunicationEfficiencyReward(BaseRewardComponent):
    """Reward for efficient agent communication."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.message_penalty = config['message_penalty']
        self.success_bonus = config['success_bonus']
        self.bandwidth_limit = config.get('bandwidth_limit', 10)

    def compute(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        if info is None:
            return 0.0

        messages_sent = info.get('messages_sent', 0)
        task_success = info.get('cooperative_task_success', False)

        comm_penalty = -self.message_penalty * max(
            0, messages_sent - self.bandwidth_limit
        )
        success_reward = self.success_bonus if task_success else 0.0

        return comm_penalty + success_reward


class AdaptiveDifficultyReward(BaseRewardComponent):
    """Dynamically adjust reward based on agent performance."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.base_reward = config['base_reward']
        self.difficulty_scale = config['difficulty_scale']
        self.success_threshold = config.get('success_threshold', 0.8)
        self.recent_successes = []
        self.window_size = config.get('window_size', 100)
        self.current_difficulty = 1.0

    def compute(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        if info is None:
            return 0.0

        task_success = info.get('task_completed', False)
        self._update_performance(task_success)
        self._adjust_difficulty()

        return self.base_reward * self.current_difficulty if task_success else 0.0

    def _update_performance(self, success: bool) -> None:
        self.recent_successes.append(float(success))
        if len(self.recent_successes) > self.window_size:
            self.recent_successes.pop(0)

    def _adjust_difficulty(self) -> None:
        if len(self.recent_successes) < self.window_size // 2:
            return

        success_rate = np.mean(self.recent_successes)
        if success_rate > self.success_threshold:
            self.current_difficulty *= 1.1
        elif success_rate < self.success_threshold * 0.5:
            self.current_difficulty *= 0.9

        self.current_difficulty = np.clip(self.current_difficulty, 0.1, 10.0)

    def reset(self) -> None:
        self.recent_successes = []
        self.current_difficulty = 1.0
