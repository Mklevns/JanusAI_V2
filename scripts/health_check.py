#!/usr/bin/env python3
"""Health check script for production deployment."""

import sys
import torch
import psutil
import requests
import os
from pathlib import Path


def check_gpu():
    """Check GPU availability and memory."""
    if not torch.cuda.is_available():
        return False, "No GPU available"

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free_memory = props.total_memory - torch.cuda.memory_allocated(i)
        if free_memory < 1e9:  # Less than 1GB free
            return False, f"GPU {i} low memory: {free_memory/1e9:.2f}GB free"

    return True, f"{torch.cuda.device_count()} GPUs available"


def check_disk_space():
    """Check available disk space."""
    usage = psutil.disk_usage('/')
    if usage.percent > 90:
        return False, f"Disk usage critical: {usage.percent}%"
    return True, f"Disk usage: {usage.percent}%"


def check_checkpoints():
    """Check if checkpoints directory is accessible."""
    checkpoint_dir = Path("/app/checkpoints")
    if not checkpoint_dir.exists():
        return False, "Checkpoint directory not found"
    if not checkpoint_dir.is_dir():
        return False, "Checkpoint path is not a directory"
    if not os.access(checkpoint_dir, os.W_OK):
        return False, "Checkpoint directory not writable"
    return True, "Checkpoint directory accessible"


def check_tensorboard():
    """Check if TensorBoard is running."""
    try:
        response = requests.get("http://localhost:6006", timeout=5)
        if response.status_code == 200:
            return True, "TensorBoard is running"
    except:
        pass
    return False, "TensorBoard not accessible"


def main():
    """Run all health checks."""
    checks = [
        ("GPU", check_gpu),
        ("Disk Space", check_disk_space),
        ("Checkpoints", check_checkpoints),
        ("TensorBoard", check_tensorboard),
    ]

    all_passed = True
    for name, check_func in checks:
        passed, message = check_func()
        status = "✓" if passed else "✗"
        print(f"{status} {name}: {message}")
        if not passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
