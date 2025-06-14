"""Base class for intent extraction from conversations."""

import abc

from ..models.conversation import Conversation
from ..models.intent import Intent


class IntentExtractor(abc.ABC):
    """Abstract base class for extracting intents from conversations.
    
    Intent extractors analyze conversation history to determine what participant
    has initiated the conversation and what their intent is.
    """
    
    @abc.abstractmethod
    async def extract_intent(self, conversation: Conversation) -> Intent | None:
        """Extract intent from a conversation.
        
        Args:
            conversation: The conversation to analyze
            
        Returns:
            Extracted intent or None if no intent could be determined
        """
        ...
