"""Base participant interface for conversation simulation."""

import abc

from ..models.conversation import Conversation
from ..models.message import Message


class Participant(abc.ABC):
    """Base interface for all simulated conversation participants.
    
    This abstract base class defines the interface that all conversation
    participants (customers and agents) must implement.
    """
    
    @abc.abstractmethod
    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate the next message based on conversation history.
        
        Args:
            conversation: The conversation history up to this point
            
        Returns:
            Message with timestamp, or None if participant is finished
        """
        ...
    
    @abc.abstractmethod
    def with_intent(self, intent_description: str) -> 'Participant':
        """Return a new participant instance with the specified intent.
        
        Args:
            intent_description: Natural language description of the participant's goal/intent
            
        Returns:
            New participant instance with the intent installed
        """
        ...


class ParticipantFactory(abc.ABC):
    """Abstract base factory for creating conversation participants.
    
    This base class provides a common interface for all participant factories,
    enabling polymorphic creation of both customer and agent participants.
    """
    
    @abc.abstractmethod
    def create_participant(self) -> Participant:
        """Create a conversation participant.
        
        Returns:
            Configured participant instance
        """
        ...
