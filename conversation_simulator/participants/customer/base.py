"""Customer participant base class and factory interface."""

import abc

from ..base import Participant
from ...models.roles import ParticipantRole


class Customer(Participant):
    """Base class for customer participants.
    
    Provides common functionality and interface for customer implementations.
    All customer participants should inherit from this class.
    """
    
    @property
    def role(self) -> ParticipantRole:
        """Customer role identifier."""
        return ParticipantRole.CUSTOMER


class CustomerFactory(abc.ABC):
    """Abstract base factory for creating customer participants."""
    
    @abc.abstractmethod
    def create_participant(self) -> Customer:
        """Create a customer participant.
            
        Returns:
            Configured customer instance
        """
        ...
