# janus/rewards/base.py
"""Base classes for the modular reward system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseRewardComponent(ABC):
    """Abstract base class for all reward components.

    Each reward component computes a single aspect of the reward signal.
    Components are designed to be modular and composable through the RewardHandler.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the reward component.

        Args:
            config: Configuration dictionary for this component.
        """
        self.config = config
        self.enabled = config.get("enabled", True)

    @abstractmethod
    def compute(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute the reward value for this component.

        Args:
            observation: Current state observation.
            action: Action taken in the current state.
            next_observation: Next state observation after taking action.
            info: Optional dictionary with additional environment information.

        Returns:
            Computed reward value for this component.
        """
        pass

    def reset(self) -> None:
        """Reset any internal state of the reward component."""
        pass


# janus/rewards/components.py
"""Concrete implementations of reward components."""

from typing import Any, Dict, Optional
import numpy as np
from .base import BaseRewardComponent


class TaskSuccessReward(BaseRewardComponent):
    """Reward component for task completion success."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize task success reward component.

        Args:
            config: Must contain 'success_reward' value.
        """
        super().__init__(config)
        self.success_reward = config["success_reward"]

    def compute(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute reward based on task success.

        Returns:
            success_reward if task completed, 0 otherwise.
        """
        if info is None:
            return 0.0

        if info.get("task_completed", False):
            return self.success_reward
        return 0.0


class ComplexityPenalty(BaseRewardComponent):
    """Penalize solutions based on their complexity."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize complexity penalty component.

        Args:
            config: Must contain 'penalty_weight' value.
        """
        super().__init__(config)
        self.penalty_weight = config["penalty_weight"]

    def compute(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute penalty based on solution complexity.

        Returns:
            Negative reward proportional to complexity.
        """
        if info is None:
            return 0.0

        complexity = info.get("solution_complexity", 0)
        return -self.penalty_weight * complexity


class NoveltyBonus(BaseRewardComponent):
    """Provide bonus reward for novel behaviors."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize novelty bonus component.

        Args:
            config: Configuration with 'bonus_scale' and 'memory_size'.
        """
        super().__init__(config)
        self.bonus_scale = config["bonus_scale"]
        self.memory_size = config.get("memory_size", 1000)
        self.state_memory = []

    def compute(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute novelty bonus based on state visitation.

        Returns:
            Bonus reward for visiting novel states.
        """
        # Compute novelty as distance to nearest neighbor
        if len(self.state_memory) == 0:
            novelty = 1.0
        else:
            distances = [
                np.linalg.norm(next_observation - mem) for mem in self.state_memory
            ]
            novelty = min(distances) / (1.0 + min(distances))

        # Update memory
        self._update_memory(next_observation)

        return self.bonus_scale * novelty

    def _update_memory(self, state: np.ndarray) -> None:
        """Update state memory with FIFO policy."""
        self.state_memory.append(state.copy())
        if len(self.state_memory) > self.memory_size:
            self.state_memory.pop(0)

    def reset(self) -> None:
        """Clear the state memory."""
        self.state_memory = []


class ProgressReward(BaseRewardComponent):
    """Reward based on progress towards goal."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize progress reward component.

        Args:
            config: Configuration with 'scale_factor'.
        """
        super().__init__(config)
        self.scale_factor = config["scale_factor"]

    def compute(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute reward based on progress metric.

        Returns:
            Scaled progress reward.
        """
        if info is None:
            return 0.0

        progress = info.get("progress", 0.0)
        return self.scale_factor * progress


# janus/rewards/handler.py
"""RewardHandler for composing multiple reward components."""

from typing import Any, Dict, List, Optional, Type
import numpy as np
from .base import BaseRewardComponent
from .components import (
    TaskSuccessReward,
    ComplexityPenalty,
    NoveltyBonus,
    ProgressReward,
)


class RewardHandler:
    """Manages and composes multiple reward components."""

    # Registry of available reward components
    COMPONENT_REGISTRY: Dict[str, Type[BaseRewardComponent]] = {
        "task_success": TaskSuccessReward,
        "complexity_penalty": ComplexityPenalty,
        "novelty_bonus": NoveltyBonus,
        "progress": ProgressReward,
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize RewardHandler from configuration.

        Args:
            config: Configuration dict with 'components' list.
        """
        self.config = config
        self.components: List[BaseRewardComponent] = []
        self.weights: List[float] = []

        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize reward components from config."""
        components_config = self.config.get("components", [])

        for comp_config in components_config:
            comp_type = comp_config["type"]
            weight = comp_config.get("weight", 1.0)

            if comp_type not in self.COMPONENT_REGISTRY:
                raise ValueError(f"Unknown component type: {comp_type}")

            component_class = self.COMPONENT_REGISTRY[comp_type]
            component = component_class(comp_config)

            if component.enabled:
                self.components.append(component)
                self.weights.append(weight)

    def compute_reward(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute total reward from all components.

        Args:
            observation: Current state observation.
            action: Action taken.
            next_observation: Next state observation.
            info: Optional environment info.

        Returns:
            Weighted sum of all component rewards.
        """
        total_reward = 0.0

        for component, weight in zip(self.components, self.weights):
            component_reward = component.compute(
                observation, action, next_observation, info
            )
            total_reward += weight * component_reward

        return total_reward

    def reset(self) -> None:
        """Reset all reward components."""
        for component in self.components:
            component.reset()

    def get_component_breakdown(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Get individual reward values from each component.

        Returns:
            Dictionary mapping component names to their rewards.
        """
        breakdown = {}

        for component, weight in zip(self.components, self.weights):
            comp_name = component.__class__.__name__
            comp_reward = component.compute(observation, action, next_observation, info)
            breakdown[comp_name] = comp_reward
            breakdown[f"{comp_name}_weighted"] = weight * comp_reward

        breakdown["total"] = self.compute_reward(
            observation, action, next_observation, info
        )

        return breakdown


# janus/rewards/__init__.py
"""Modular reward system for Janus AI."""

from .base import BaseRewardComponent
from .handler import RewardHandler
from .components import (
    TaskSuccessReward,
    ComplexityPenalty,
    NoveltyBonus,
    ProgressReward,
)

__all__ = [
    "BaseRewardComponent",
    "RewardHandler",
    "TaskSuccessReward",
    "ComplexityPenalty",
    "NoveltyBonus",
    "ProgressReward",
]


# Example config.yaml section
"""
# Add this to your config.yaml file:

rewards:
  components:
    - type: task_success
      enabled: true
      weight: 10.0
      success_reward: 100.0
      
    - type: complexity_penalty
      enabled: true
      weight: 0.1
      penalty_weight: 1.0
      
    - type: novelty_bonus
      enabled: true
      weight: 0.5
      bonus_scale: 2.0
      memory_size: 1000
      
    - type: progress
      enabled: true
      weight: 1.0
      scale_factor: 5.0
"""


# Integration example for PPOTrainer
# Example: In your PPOTrainer, integrate the RewardHandler:
#
# from janus.rewards import RewardHandler
#
# class PPOTrainer:
#     def __init__(self, config: Dict[str, Any]):
#         # ... existing initialization ...
#
#         # Initialize reward handler
#         reward_config = config.get('rewards', {})
#         self.reward_handler = RewardHandler(reward_config)
#
#     def collect_rollouts(self):
#         # ... existing rollout collection ...
#
#         # Replace direct reward calculation with:
#         reward = self.reward_handler.compute_reward(
#             observation=obs,
#             action=action,
#             next_observation=next_obs,
#             info=info
#         )
#
#         # Optional: log component breakdown for debugging
#         if self.debug_rewards:
#             breakdown = self.reward_handler.get_component_breakdown(
#                 obs, action, next_obs, info
#             )
#             self.logger.debug(f"Reward breakdown: {breakdown}")
