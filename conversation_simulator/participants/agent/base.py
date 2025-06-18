"""Agent participant base class and factory interface."""

import abc

from ..base import Participant, ParticipantFactory
from ...models.roles import ParticipantRole


class Agent(Participant):
    """Base class for agent participants.
    
    Provides common functionality and interface for agent implementations.
    All agent participants should inherit from this class.
    """
    
    @property
    def role(self) -> ParticipantRole:
        """Agent role identifier."""
        return ParticipantRole.AGENT


class AgentFactory(ParticipantFactory):
    """Abstract base factory for creating agent participants."""
    
    @abc.abstractmethod
    def create_participant(self) -> Agent:
        """Create an agent participant.
            
        Returns:
            Configured agent instance
        """
        ...
