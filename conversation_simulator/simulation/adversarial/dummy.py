"""Module providing a dummy implementation of adversarial testing.

This module contains a simple implementation of an adversarial tester that always fails
to identify the real conversation, making it useful as a baseline or for testing.
"""

from typing import override
import attrs

from conversation_simulator.models.conversation import Conversation
from .base import AdversarialTester


@attrs.frozen
class DummyAdversarialTester(AdversarialTester):
    """A dummy adversarial tester that always returns False.
    
    Temporarily used for intergration testing or as a placeholder.
    """
    
    @override
    async def identify_real_conversation(
        self,
        real_conversation: Conversation,
        simulated_conversation: Conversation,
    ) -> bool:
        """Always returns False, indicating failure to identify the real conversation.
        
        Args:
            real_conversation: The actual conversation from real data
            simulated_conversation: The AI-generated simulated conversation
            
        Returns:
            bool: Always False, indicating this tester cannot distinguish between
                real and simulated conversations
        """
        return False
