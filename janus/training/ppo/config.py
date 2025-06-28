# janus/training/ppo/config.py
"""
Configuration module for the PPO Trainer using Pydantic for robust validation.
"""

import yaml
import logging
import os
import math # Added import
import re
import sys
import datetime
import torch
from pathlib import Path
from typing import Optional, Union, Dict, Any
from pydantic import BaseModel, validator, Field

logger = logging.getLogger(__name__)


def _substitute_env_vars(yaml_str: str) -> str:
    """
    Substitutes environment variables in the YAML string.
    e.g., ${VAR_NAME:default_value}
    """
    pattern = re.compile(r"\$\$\{([^}]+)\}")

    def replacer(match):
        key_default = match.group(1).split(":", 1)
        key = key_default[0]
        default = key_default[1] if len(key_default) > 1 else ""
        return os.environ.get(key, default)

    return pattern.sub(replacer, yaml_str)


class PPOConfig(BaseModel):
    """
    A Pydantic-based, validated configuration schema for the PPO algorithm.
    Defines and validates all hyperparameters and execution settings.
    """

    # Core PPO hyperparameters
    learning_rate: float = Field(
        3e-4, gt=0, le=1, description="Initial learning rate for the Adam optimizer."
    )
    gamma: float = Field(
        0.99, gt=0, le=1, description="Discount factor for future rewards."
    )
    gae_lambda: float = Field(
        0.95, gt=0, le=1, description="Lambda for Generalized Advantage Estimation."
    )
    clip_epsilon: float = Field(0.2, gt=0, le=1, description="PPO clipping parameter.") # Changed lt=1 to le=1
    value_coef: float = Field(
        0.5, ge=0, description="Coefficient for the value function loss."
    )
    entropy_coef: float = Field(
        0.01, ge=0, description="Coefficient for the entropy bonus."
    )
    n_epochs: int = Field(
        10, gt=0, description="Number of epochs to train on collected data per update."
    )
    batch_size: int = Field(64, gt=0, description="Minibatch size for policy updates.")
    action_space_type: str = Field(
        "discrete",
        pattern=r"^(discrete|continuous)$",
        description="Type of action space.",
    )

    # Stability and optimization
    max_grad_norm: float = Field(
        0.5, gt=0, description="Maximum norm for gradient clipping."
    )
    use_mixed_precision: bool = Field(
        False, description="Enable automatic mixed precision training (AMP)."
    )
    gradient_accumulation_steps: int = Field(
        1, ge=1, description="Number of steps to accumulate gradients before updating."
    )
    clip_vloss: bool = Field(
        True, description="Whether to use a clipped value function loss."
    )

    # Hyperparameter scheduling
    lr_schedule: str = Field(
        "constant",
        pattern=r"^(constant|linear|cosine)$",
        description="Learning rate schedule.",
    )
    clip_schedule: str = Field(
        "constant",
        pattern=r"^(constant|linear|cosine)$",
        description="Clipping epsilon schedule.",
    )
    entropy_schedule: str = Field(
        "constant",
        pattern=r"^(constant|linear|cosine)$",
        description="Entropy coefficient schedule.",
    )
    lr_end: float = Field(1e-5, ge=0, description="Final learning rate for scheduling.")
    clip_end: float = Field(
        0.1, ge=0, description="Final clipping epsilon for scheduling."
    )
    entropy_end: float = Field(
        0.001, ge=0, description="Final entropy coefficient for scheduling."
    )

    # Early stopping
    target_kl: Optional[float] = Field(
        0.015, gt=0, description="Target KL divergence for early stopping."
    )

    # Normalization
    normalize_advantages: bool = Field(
        True, description="Whether to normalize advantages."
    )
    normalize_rewards: bool = Field(
        False, description="Whether to normalize rewards with a running mean/std."
    )

    # Hardware and Execution
    device: str = Field(
        "auto", description="Device to run training on ('cuda', 'cpu', 'auto')."
    )
    num_workers: int = Field(
        4, ge=1, description="Number of parallel workers for environment interaction."
    )
    use_subprocess_envs: bool = Field(
        True, description="Use subprocesses for parallel envs to bypass GIL."
    )

    # Error handling
    fail_on_env_error: bool = Field(
        False,
        description="Whether to crash on an environment error or attempt to recover.",
    )
    max_env_retries: int = Field(
        3, ge=0, description="Maximum number of retries for a failing environment step."
    )

    # Logging and Experiment Tracking
    log_interval: int = Field(10, ge=1, description="Log metrics every N updates.")
    save_interval: int = Field(
        100, ge=1, description="Save a checkpoint every N updates."
    )
    eval_interval: int = Field(
        1000, ge=1, description="Evaluate the agent every N updates."
    )
    experiment_tags: Dict[str, Any] = Field(
        default_factory=dict, description="Tags for experiment tracking (e.g., in W&B)."
    )
    experiment_name: str = "ppo_experiment"

    # New fields for production support (as per prompt)
    enable_profiling: bool = Field(
        False, description="Enable PyTorch profiler for performance analysis"
    )
    profiler_trace_path: str = Field(
        "traces/", description="Path to save profiler traces"
    )
    enable_anomaly_detection: bool = Field(
        False, description="Enable PyTorch anomaly detection (slow!)"
    )
    checkpoint_metric: str = Field(
        "mean_episode_reward",
        description="Metric to use for best checkpoint selection"
    )

    class Config:
        """Pydantic config class to forbid extra fields."""

        extra = "forbid"

    # As per prompt, a validator for learning_rate, gamma, gae_lambda, clip_epsilon
    # to be in (0, 1]. The Field definitions already cover this (gt=0, le=1).
    # If a single validator function is desired, it can be implemented like this:
    @validator("learning_rate", "gamma", "gae_lambda", "clip_epsilon", always=True)
    def validate_positive_float_in_range(cls, v, field):
        """Ensure critical hyperparameters are in valid (0, 1] range."""
        if not (0 < v <= 1):
            # This should ideally not be hit if Field definitions are correct
            # but serves as an additional check or if Field defs were less strict.
            raise ValueError(f"{field.name} must be in (0, 1], got {v}")
        return v

    @validator("batch_size")
    def validate_batch_size(cls, v):
        """Ensure batch size is power of 2 for GPU efficiency."""
        if v <= 0: # batch_size must be positive, Field(gt=0) already ensures this.
             raise ValueError(f"batch_size must be positive, got {v}")
        if v & (v - 1) != 0: # Check if not a power of 2
            # Round up to next power of 2
            new_v = 2 ** math.ceil(math.log2(v))
            logger.warning(
                "Batch size %d is not a power of 2. Rounding to %d for GPU efficiency.",
                v, new_v
            )
            return new_v
        return v

    @validator("device", pre=True, always=True)
    def set_device_auto(cls, v):
        """Automatically selects device if set to 'auto'."""
        if v == "auto":
            try:
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return v

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PPOConfig":
        """Loads configuration from a YAML file, substituting environment variables."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                yaml_str = f.read()
        except FileNotFoundError:
            logger.error("Configuration file not found at: %s", path)
            raise

        yaml_str = _substitute_env_vars(yaml_str)
        data = yaml.safe_load(yaml_str)
        return cls(**data)

    def to_yaml(self, path: Union[str, Path], include_metadata: bool = True):
        """Saves the configuration to a YAML file."""
        config_dict = self.dict()

        if include_metadata:
            version_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            config_dict["_metadata"] = {
                "experiment_name": self.experiment_name,
                "timestamp": datetime.datetime.now().isoformat(),
                "python_version": version_str,
            }

        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            logger.info("Configuration saved to %s", path)
        except IOError as e:
            logger.error("Failed to save configuration to %s: %s", path, e)

    def __str__(self):
        """String representation of the configuration."""
        return yaml.dump(self.dict(), default_flow_style=False, sort_keys=False)
