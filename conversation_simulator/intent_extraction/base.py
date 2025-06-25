"""Base class for intent extraction from conversations."""

import abc

from ..models.conversation import Conversation
from ..models.intent import Intent


class IntentExtractor(abc.ABC):
    """Abstract base class for extracting intents from conversations.
    
    Intent extractors analyze conversation history to determine what participant
    has initiated the conversation and what their intent is.
    """
    
    async def extract_intent(self, conversation: Conversation) -> Intent | None:
        """Extract intent from a conversation.
        
        Args:
            conversation: The conversation to analyze
            
        Returns:
            Extracted intent or None if no intent could be determined
        """
        return (await self.extract_intents((conversation, )))[0]

    @abc.abstractmethod
    async def extract_intents(self, conversations: tuple[Conversation, ...]) -> tuple[Intent | None, ...]:
        """Extract intents from a conversation. See extract_intent for details.
        """
        ...
