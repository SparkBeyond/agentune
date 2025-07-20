"""Real agent participant implementations.

This module provides base classes for integrating real (external) agent systems
with the conversation simulation framework.
"""

from .real import RealAgent, RealAgentFactory

__all__ = ["RealAgent", "RealAgentFactory"]