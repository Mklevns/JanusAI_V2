# File: janus/setup.py
"""Setup script for Janus PPO package.
This script defines the package metadata, dependencies,
and entry points for the Janus PPO implementation.
It uses setuptools for packaging and distribution.
"""

import os
from setuptools import setup, find_packages
# Ensure the script is run from the package root
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
try:
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = "Production-ready PPO implementation for JanusAI V2."

setup(
    name="janus-ppo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.3",
        "pydantic>=2.5.0",
        "PyYAML>=6.0.1",
        "tensorboard>=2.15.0",
        "psutil>=5.9.0",  # Always needed for CPU/memory monitoring
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "mypy>=1.7.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "pre-commit>=3.5.0",
        ],
        "gpu": [
            "gputil>=1.4.0",  # GPU monitoring
            "nvitop>=1.3.0",  # Alternative GPU monitor
        ],
        "full": [
            "wandb>=0.16.0",
            "gymnasium>=0.29.1",
            "matplotlib>=3.8.2",
            "tqdm>=4.66.1",
            "rich>=13.7.0",
        ],
        "profiling": [
            "line-profiler>=4.1.1",
            "memory-profiler>=0.61.0",
            "scalene>=1.5.31",
        ],
    },
    author="Your Name",
    description="Production-ready PPO implementation for JanusAI V2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
