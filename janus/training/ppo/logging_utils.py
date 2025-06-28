# janus/training/ppo/logging_utils.py

'''
Logging utilities for Janus PPO training.
'''

import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging(experiment_name, checkpoint_dir, use_tensorboard=True, use_wandb=False, config_dict=None):
    """
    Set up logging infrastructure for the training process.

    Args:
        experiment_name (str): Name of the experiment for organizing logs.
        checkpoint_dir (Path): Directory to store logs and checkpoints.
        use_tensorboard (bool): Whether to use TensorBoard logging.
        use_wandb (bool): Whether to use Weights & Biases logging.
        config_dict (dict): Configuration dictionary to log to W&B.

    Returns:
        dict: Contains TensorBoard writer, CSV logger, and optional W&B setup.
    """
    log_dir = checkpoint_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir) if use_tensorboard else None

    # Setup CSV logger
    csv_log_path = log_dir / "training_log.csv"
    csv_logger = logging.getLogger("csv_logger")
    csv_logger.setLevel(logging.INFO)
    csv_handler = logging.FileHandler(csv_log_path, mode='a', encoding='utf-8')
    csv_handler.setFormatter(logging.Formatter('%(message)s'))
    if csv_logger.hasHandlers():
        csv_logger.handlers.clear()
    csv_logger.addHandler(csv_handler)

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project=experiment_name, config=config_dict or {})

    return {
        "writer": writer,
        "csv": csv_logger
    }
