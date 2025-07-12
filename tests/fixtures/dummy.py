"""Dummy implementations for testing purposes.

This module contains simple dummy implementations that serve as placeholders
or baseline implementations for testing and development.
"""

from __future__ import annotations
from typing import override
import attrs

from conversation_simulator.intent_extraction.base import IntentExtractor
from conversation_simulator.models.conversation import Conversation
from conversation_simulator.models.intent import Intent
from conversation_simulator.simulation.adversarial.base import AdversarialTest, AdversarialTester


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


@attrs.frozen
class DummyAdversarialTester(AdversarialTester):
    """A dummy adversarial tester that always returns False.
    
    Temporarily used for integration testing or as a placeholder.
    """
    
    @override
    async def identify_real_conversations(
        self,
        instances: tuple[AdversarialTest, ...],
        return_exceptions: bool = True
    ) -> tuple[bool | None | Exception, ...]:
        """Always returns False, indicating this tester cannot distinguish between real and simulated conversations
        """
        return (False, ) * len(instances)