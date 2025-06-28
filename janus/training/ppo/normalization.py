# janus/training/ppo/normalization.py

'''
Normalization utilities for PPO training in JanusAI V2.
This module provides a RunningMeanStd class for maintaining running statistics
of observations, which is crucial for stabilizing training by normalizing inputs.'''

import numpy as np
import threading
from typing import Tuple

class RunningMeanStd:
    """Numerically stable running statistics with Welford's algorithm."""

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()): # Default shape to scalar
        """Initialize with better numerical stability."""
        if not isinstance(shape, tuple): # Ensure shape is a tuple
            shape = (shape,) if isinstance(shape, (int, np.integer)) else ()

        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)  # Ensure count is float
        self.lock = threading.Lock()
        self.m2 = np.zeros(shape, dtype=np.float64)  # For Welford's algorithm
        self._epsilon = float(epsilon) # Store for potential reset

    def update(self, x: np.ndarray) -> None:
        """Update statistics using Welford's online algorithm.
        x is expected to be a batch of samples, i.e., x.shape = (batch_size, *sample_shape).
        sample_shape must match self.mean.shape.
        """

        # If x is a single sample (its shape matches self.mean.shape), add a batch dimension.
        if x.shape == self.mean.shape:
            x = x.reshape(1, *self.mean.shape)

        # After potential reshaping, x must be at least 2D (batch_dim, feature_dims...)
        # And x.shape[1:] must match self.mean.shape
        if x.ndim < 1 or x.shape[1:] != self.mean.shape: # ndim < 1 is for empty x
             # Allow for scalar mean and x being (N, 1)
            if not (self.mean.shape == () and x.ndim == 2 and x.shape[1] == 1):
                raise ValueError(
                    f"Input x has shape {x.shape}. Expected (batch_size, *sample_shape) "
                    f"where sample_shape is {self.mean.shape}."
                )

        with self.lock:
            batch_size = x.shape[0]

            for i in range(batch_size):
                self.count += 1.0
                data_point = x[i]

                # If mean is scalar, data_point (which is x[i]) might be (1,)
                if self.mean.shape == () and data_point.shape == (1,):
                    data_point = data_point.item() # Convert to scalar

                # data_point should match self.mean.shape at this point
                delta = data_point - self.mean
                self.mean += delta / self.count
                delta2 = data_point - self.mean
                self.m2 += delta * delta2

            if self.count > 1.0:
                self.var = self.m2 / (self.count - 1.0)
            else:
                self.var = np.ones_like(self.var)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input x with current statistics."""
        with self.lock:
            return (x - self.mean) / np.sqrt(self.var + 1e-8) # 1e-8 for numerical stability