# janus/rewards/reward_handler.py
"""
RewardHandler module for managing and composing multiple reward components.
This module supports both core and custom-defined reward components, allowing 
for flexible reward structures in reinforcement learning environments.
"""

from typing import Any, Dict, List, Optional, Type
from .base import BaseRewardComponent
from .components import (
    TaskSuccessReward,
    ComplexityPenalty,
    NoveltyBonus,
    ProgressReward,
)
from .custom_components import (
    SymbolicRegressionReward,
    CommunicationEfficiencyReward,
    AdaptiveDifficultyReward,
)


class RewardHandler:
    """
    Manages and composes multiple reward components based on config.
    Automatically supports both core and custom-defined reward modules.
    """

    COMPONENT_REGISTRY: Dict[str, Type[BaseRewardComponent]] = {
        "task_success": TaskSuccessReward,
        "complexity_penalty": ComplexityPenalty,
        "novelty_bonus": NoveltyBonus,
        "progress": ProgressReward,
        "symbolic_regression": SymbolicRegressionReward,
        "communication_efficiency": CommunicationEfficiencyReward,
        "adaptive_difficulty": AdaptiveDifficultyReward,
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize RewardHandler from configuration.

        Args:
            config: Dictionary containing 'components' list.
        """
        self.config = config
        self.components: List[BaseRewardComponent] = []
        self.weights: List[float] = []
        self._initialize_components()

    def _initialize_components(self) -> None:
        components_config = self.config.get("components", [])
        for comp_config in components_config:
            comp_type = comp_config["type"]
            weight = comp_config.get("weight", 1.0)

            if comp_type not in self.COMPONENT_REGISTRY:
                raise ValueError(f"Unknown reward component type: {comp_type}")

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
        """Compute weighted sum of all reward components."""
        total_reward = 0.0
        for component, weight in zip(self.components, self.weights):
            reward = component.compute(observation, action, next_observation, info)
            total_reward += weight * reward
        return total_reward

    def get_component_breakdown(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Return detailed reward breakdown per component."""
        breakdown = {}
        for component, weight in zip(self.components, self.weights):
            name = component.__class__.__name__
            value = component.compute(observation, action, next_observation, info)
            breakdown[name] = value
            breakdown[f"{name}_weighted"] = weight * value

        breakdown["total"] = self.compute_reward(
            observation, action, next_observation, info
        )
        return breakdown

    def reset(self) -> None:
        """Reset all reward components."""
        for component in self.components:
            component.reset()
