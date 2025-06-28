import psutil
import GPUtil
from typing import Dict, List, Optional
import time
import logging

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor system resources during training."""

    def __init__(self, log_interval: int = 60):
        """Initialize system monitor."""
        self.log_interval = log_interval
        self.last_log_time = time.time()
        self.gpu_available = len(GPUtil.getGPUs()) > 0

    def get_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_gb": psutil.virtual_memory().used / (1024**3),
        }

        if self.gpu_available:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                metrics[f"gpu_{i}_utilization"] = gpu.load * 100
                metrics[f"gpu_{i}_memory_percent"] = gpu.memoryUtil * 100
                metrics[f"gpu_{i}_temperature"] = gpu.temperature

        return metrics

    def log_if_needed(self, writer=None, step: int = 0) -> None:
        """Log metrics if enough time has passed."""
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            metrics = self.get_metrics()

            # Log to console
            logger.info("System metrics: %s", metrics)

            # Log to TensorBoard if available
            if writer:
                for name, value in metrics.items():
                    writer.add_scalar(f"system/{name}", value, step)

            self.last_log_time = current_time
