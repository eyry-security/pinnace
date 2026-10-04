"""Pinnace — a general multi-turn agent runtime.

The engine of the Eyry recon suite: hand it a prompt or seed and it runs an
LLM-driven agent loop with tool use, context compaction, and execution inside a
Docker sandbox. Used standalone, or driven by Aplomado and Quarterdeck.
"""

from .agent import Agent
from .config import AgentConfig

__version__ = "0.1.0"
__all__ = ["Agent", "AgentConfig", "__version__"]
