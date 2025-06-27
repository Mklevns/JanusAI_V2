from setuptools import setup, find_packages

setup(
    name="janus",
    version="0.1.0",
    description="Symbolic knowledge discovery with multi-agent RL.",
    author="JanusAI Team",
    packages=find_packages(),
    install_requires=[
        "torch",
        "gymnasium",
        "sympy",
        "pyyaml",
        "wandb",
        "typer",
    ],
)
