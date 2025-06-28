# janus/training/ppo/logging_utils.py

'''
Logging utilities for Janus PPO training.
'''

import logging
from pathlib import Path
from typing import Optional, Dict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging(name: str, checkpoint_dir: Path, use_tensorboard: bool, use_wandb: bool, config: dict) -> Dict[str, Optional[object]]:
    loggers = {}

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if use_tensorboard:
        loggers["writer"] = SummaryWriter(log_dir=f"runs/{name}")
    else:
        loggers["writer"] = None

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project="janus-ppo", name=name, config=config)
        loggers["wandb"] = wandb
    else:
        loggers["wandb"] = None

    csv_logger = logging.getLogger(f"{name}_csv")
    csv_handler = logging.FileHandler(checkpoint_dir / "metrics.csv")
    csv_handler.setFormatter(logging.Formatter("%(message)s"))
    csv_logger.addHandler(csv_handler)
    csv_logger.setLevel(logging.INFO)
    csv_logger.info("update,global_step,mean_reward,policy_loss,value_loss,entropy,kl_div,learning_rate")
    loggers["csv"] = csv_logger

    return loggers
