"""Module providing a dummy implementation of adversarial testing.

This module contains a simple implementation of an adversarial tester that always fails
to identify the real conversation, making it useful as a baseline or for testing.
"""

from typing import override
import attrs

from .base import AdversarialTest, AdversarialTester


@attrs.frozen
class DummyAdversarialTester(AdversarialTester):
    """A dummy adversarial tester that always returns False.
    
    Temporarily used for intergration testing or as a placeholder.
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
