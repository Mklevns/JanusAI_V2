# janus/training/ppo/__init__.py
"""
Proximal Policy Optimization (PPO) training modules.

This package provides a modular, production-ready implementation of PPO
with support for standard training and World Model-based training.
"""

from .config import PPOConfig
from .trainer import PPOTrainer
from .buffer import RolloutBuffer
from .collector import AsyncRolloutCollector
from .normalization import RunningMeanStd
from .world_model_trainer import WorldModelPPOTrainer, WorldModelConfig

__all__ = [
    'PPOConfig',
    'PPOTrainer',
    'WorldModelPPOTrainer',
    'WorldModelConfig',
    'RolloutBuffer',
    'AsyncRolloutCollector',
    'RunningMeanStd'
]
