"""Dummy intent extractor implementation for testing and development."""

from __future__ import annotations

from .base import IntentExtractor
from ..models.conversation import Conversation
from ..models.intent import Intent


class DummyIntentExtractor(IntentExtractor):
    """Dummy intent extractor that uses the first message as the intent.
    
    This is a simple implementation for testing and development purposes.
    It extracts the intent by taking the text of the first message in
    the conversation and treating it as the intent description.
    """
    
    async def extract_intent(self, conversation: Conversation) -> Intent | None:
        """Extract intent from the first message in the conversation.
        
        Args:
            conversation: The conversation to analyze
            
        Returns:
            Intent based on first message, or None if conversation is empty
        """
        if conversation.is_empty:
            return None
        
        first_message = conversation.messages[0]
        
        # Create intent using the first message's content as description
        # and the sender's role
        return Intent(
            role=first_message.sender,
            description=first_message.content
        )
