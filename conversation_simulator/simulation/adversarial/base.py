"""Base class for adversarial testing of conversation simulation quality.

Adversarial testing evaluates how well simulated conversations can be distinguished
from real conversations by testing whether a model or human can identify which
conversation is real when presented with a pair.
"""
from abc import ABC, abstractmethod
import asyncio

from ...models.conversation import Conversation


class AdversarialTester(ABC):
    """Base class for adversarial testing of conversation quality.
    
    Adversarial testing presents pairs of conversations (one real, one simulated)
    to an evaluator and measures how accurately the evaluator can identify the
    real conversation. Higher accuracy indicates that simulated conversations
    are easily distinguishable from real ones, while lower accuracy (closer to
    random chance at 50%) indicates higher quality simulation.
    """
    
    @abstractmethod
    async def identify_real_conversation(
        self,
        real_conversation: Conversation,
        simulated_conversation: Conversation
    ) -> bool:
        """Evaluate a single pair of conversations to determine if the real conversation
        can be correctly identified.

        Args:  
            real_conversation: The real conversation to evaluate
            simulated_conversation: The simulated conversation to evaluate
        Returns:
            bool: True if the real conversation is correctly identified, False otherwise 
        """
        ...

    async def identify_real_conversations(
        self,
        real_conversations: tuple[Conversation, ...],
        simulated_conversations: tuple[Conversation, ...],
    ) -> tuple[bool, ...]:
        """Evaluate multiple pairs of conversations concurrently.
        Args:
            real_conversations: Tuple of real conversations to evaluate
            simulated_conversations: Tuple of simulated conversations to evaluate
        Returns:
            tuple[bool, ...]: Tuple of booleans indicating whether each pair was
            correctly identified (True for correct identification, False otherwise)
        """
        # default implementation that can be overridden
        if len(real_conversations) != len(simulated_conversations):
            raise ValueError("Real and simulated conversations must have the same length")
        results = await asyncio.gather(
            *[
                self.identify_real_conversation(real, simulated)
                for real, simulated in zip(real_conversations, simulated_conversations)
            ]
        )
        return tuple(results)
