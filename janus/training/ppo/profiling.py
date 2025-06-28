import torch.profiler
from contextlib import contextmanager
from pathlib import Path


class PerformanceProfiler:
    """PyTorch profiler wrapper for PPO training."""

    def __init__(self, enabled: bool = False, trace_dir: str = "traces/"):
        """Initialize profiler."""
        self.enabled = enabled
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(exist_ok=True)
        self.profiler = None

    @contextmanager
    def profile(self, name: str = "ppo_training"):
        """Context manager for profiling."""
        if not self.enabled:
            yield
            return

        schedule = torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2
        )

        self.profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(self.trace_dir / name)
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )

        self.profiler.start()
        try:
            yield self.profiler
        finally:
            self.profiler.stop()
