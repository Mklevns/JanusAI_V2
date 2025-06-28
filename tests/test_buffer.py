import pytest
import numpy as np
import torch
import threading
from janus.training.ppo.buffer import RolloutBuffer


class TestRolloutBuffer:
    """Comprehensive tests for RolloutBuffer."""

    @pytest.fixture
    def buffer(self):
        """Create a test buffer."""
        return RolloutBuffer(
            buffer_size=10,
            obs_shape=(4,),
            action_shape=(2,),
            device=torch.device("cpu"),
            n_envs=2
        )

    def test_initialization(self, buffer):
        """Test buffer initialization."""
        assert buffer.buffer_size == 10
        assert buffer.obs_shape == (4,)
        assert buffer.action_shape == (2,)
        assert buffer.n_envs == 2
        assert buffer.ptr == 0

    def test_add_and_get(self, buffer):
        """Test adding and retrieving data."""
        # Create test data
        obs = np.random.randn(2, 4).astype(np.float32)
        actions = np.random.randn(2, 2).astype(np.float32)
        rewards = np.array([1.0, -1.0], dtype=np.float32)
        dones = np.array([False, True], dtype=np.float32)
        values = np.array([0.5, 0.3], dtype=np.float32)
        log_probs = np.array([-0.5, -0.7], dtype=np.float32)

        # Add data
        buffer.add(obs, actions, rewards, dones, values, log_probs)

        # Retrieve data
        data = buffer.get()

        assert data["observations"].shape == (1, 2, 4)
        assert data["actions"].shape == (1, 2, 2)
        assert data["rewards"].shape == (1, 2)
        assert torch.allclose(data["rewards"], torch.tensor([[1.0, -1.0]]))

    def test_buffer_overflow(self, buffer):
        """Test buffer overflow handling."""
        # Fill buffer beyond capacity
        for i in range(12):
            obs = np.random.randn(2, 4).astype(np.float32)
            actions = np.random.randn(2, 2).astype(np.float32)
            rewards = np.ones(2, dtype=np.float32) * i
            dones = np.zeros(2, dtype=np.float32)
            values = np.ones(2, dtype=np.float32) * 0.5
            log_probs = np.ones(2, dtype=np.float32) * -0.5

            buffer.add(obs, actions, rewards, dones, values, log_probs)

        # Should wrap around
        assert buffer.ptr == 2
        assert buffer.full

    def test_thread_safety(self, buffer):
        """Test concurrent access."""
        errors = []

        def add_data(thread_id):
            try:
                for _ in range(5):
                    obs = np.random.randn(2, 4).astype(np.float32)
                    actions = np.random.randn(2, 2).astype(np.float32)
                    rewards = np.ones(2, dtype=np.float32) * thread_id
                    dones = np.zeros(2, dtype=np.float32)
                    values = np.ones(2, dtype=np.float32) * 0.5
                    log_probs = np.ones(2, dtype=np.float32) * -0.5

                    buffer.add(obs, actions, rewards, dones, values, log_probs)
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = []
        for i in range(4):
            t = threading.Thread(target=add_data, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_shape_validation(self, buffer):
        """Test shape validation in add method."""
        # Wrong observation shape
        with pytest.raises(ValueError, match="shape mismatch"):
            buffer.add(
                obs=np.zeros((2, 5)),  # Wrong shape
                action=np.zeros((2, 2)),
                reward=np.zeros(2),
                done=np.zeros(2),
                value=np.zeros(2),
                log_prob=np.zeros(2),
            )
