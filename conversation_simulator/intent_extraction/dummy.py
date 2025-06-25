"""Dummy intent extractor implementation for testing and development."""

from __future__ import annotations
from typing import override

from .base import IntentExtractor
from ..models.conversation import Conversation
from ..models.intent import Intent


class DummyIntentExtractor(IntentExtractor):
    """Dummy intent extractor that uses the first message as the intent.
    
    This is a simple implementation for testing and development purposes.
    It extracts the intent by taking the text of the first message in
    the conversation and treating it as the intent description.
    """
    
    @override
    async def extract_intents(self, conversations: tuple[Conversation, ...], 
                              return_exceptions: bool = True) -> tuple[Intent | None | Exception, ...]:
        return tuple(
            Intent(
                role=conversation.messages[0].sender,
                description=conversation.messages[0].content
            ) if conversation.messages else None
            for conversation in conversations
        )

