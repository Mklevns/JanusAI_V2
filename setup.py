from setuptools import setup, find_packages

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
        "monitoring": [
            "psutil>=5.9.0",
            "gputil>=1.4.0",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    description="Production-ready PPO implementation for JanusAI V2",
)
