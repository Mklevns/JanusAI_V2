# Structure overview (we'll build each in parts):
# - config.py: PPOConfig class
# - logging_utils.py: setup logging and metrics
# - normalization.py: RunningMeanStd
# - buffer.py: RolloutBuffer
# - collector.py: AsyncRolloutCollector
# - trainer.py: PPOTrainer
# - main.py: training entrypoint with YAML loading

# Let's begin with config.py

import yaml
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Union, Dict, Any
import os
import re
import sys

logger = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    n_epochs: int = 10
    batch_size: int = 64
    max_grad_norm: float = 0.5
    use_mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    lr_schedule: str = "constant"
    clip_schedule: str = "constant"
    entropy_schedule: str = "constant"
    lr_end: float = 1e-5
    clip_end: float = 0.1
    entropy_end: float = 0.001
    target_kl: Optional[float] = 0.015
    normalize_advantages: bool = True
    normalize_rewards: bool = False
    device: str = "auto"
    num_workers: int = 4
    use_subprocess_envs: bool = True
    fail_on_env_error: bool = False
    max_env_retries: int = 3
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 1000
    experiment_tags: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PPOConfig":
        with open(path, "r") as f:
            yaml_str = f.read()

        pattern = re.compile(r"\$\{([^}]+)\}")

        def replacer(match):
            key_default = match.group(1).split(":", 1)
            key = key_default[0]
            default = key_default[1] if len(key_default) > 1 else ""
            return os.environ.get(key, default)

        yaml_str = pattern.sub(replacer, yaml_str)
        data = yaml.safe_load(yaml_str)
        return cls(**data)

    def to_yaml(self, path: Union[str, Path], include_metadata: bool = True):
        import datetime
        config_dict = asdict(self)
        if include_metadata:
            config_dict["_metadata"] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            }
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
