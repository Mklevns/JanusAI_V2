from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base callback class."""

    @abstractmethod
    def on_training_start(self, trainer: "PPOTrainer") -> None:
        """Called at the start of training."""
        pass

    @abstractmethod
    def on_rollout_end(self, trainer: "PPOTrainer", metrics: Dict[str, float]) -> None:
        """Called after rollout collection."""
        pass

    @abstractmethod
    def on_training_end(self, trainer: "PPOTrainer") -> None:
        """Called at the end of training."""
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping based on reward."""

    def __init__(self, patience: int = 50, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = -np.inf
        self.patience_counter = 0

    def on_training_start(self, trainer: "PPOTrainer") -> None:
        """Reset counters."""
        self.best_reward = -np.inf
        self.patience_counter = 0

    def on_rollout_end(self, trainer: "PPOTrainer", metrics: Dict[str, float]) -> None:
        """Check for improvement."""
        current_reward = metrics.get("mean_episode_reward", -np.inf)

        if current_reward > self.best_reward + self.min_delta:
            self.best_reward = current_reward
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            logger.info(
                "Early stopping triggered. No improvement for %d updates.",
                self.patience
            )
            trainer.should_stop = True

    def on_training_end(self, trainer: "PPOTrainer") -> None:
        """Log final stats."""
        logger.info("Training ended. Best reward: %.2f", self.best_reward)
