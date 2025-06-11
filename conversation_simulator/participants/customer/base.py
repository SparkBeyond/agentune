"""Customer participant base class."""

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
