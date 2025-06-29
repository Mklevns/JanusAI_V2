# janus/agents/components/__init__.py
"""
World Model components for agent architectures.

These components enable agents to learn internal models of their
environment and use them for planning and imagination.
"""

from .vae import VariationalAutoencoder
from .mdn_rnn import MDNRNN

__all__ = [
    'VariationalAutoencoder',
    'MDNRNN'
]
