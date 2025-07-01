"""Base class for adversarial testing of conversation simulation quality.

Adversarial testing evaluates how well simulated conversations can be distinguished
from real conversations by testing whether a model or human can identify which
conversation is real when presented with a pair.
"""
from abc import ABC, abstractmethod
from typing import cast

from attrs import frozen

from ...models.conversation import Conversation

@frozen
class AdversarialTest:
    real_conversation: Conversation
    simulated_conversation: Conversation

class AdversarialTester(ABC):
    """Base class for adversarial testing of conversation quality.
    
    Adversarial testing presents pairs of conversations (one real, one simulated)
    to an evaluator and measures how accurately the evaluator can identify the
    real conversation. Higher accuracy indicates that simulated conversations
    are easily distinguishable from real ones, while lower accuracy (closer to
    random chance at 50%) indicates higher quality simulation.
    """
    
    async def identify_real_conversation(
        self,
        real_conversation: Conversation,
        simulated_conversation: Conversation
    ) -> bool | None:
        """Evaluate a single pair of conversations to determine if the real conversation
        can be correctly identified.

        Args:  
            real_conversation: The real conversation to evaluate
            simulated_conversation: The simulated conversation to evaluate
        Returns:
            bool | None: True if the real conversation is correctly identified, False if not,
                         or None if the conversation was empty. Underlying errors
                         such as connection errors are propagated. Note that None is a valid outcome,
                         not an error.
        """
        result = (await self.identify_real_conversations(
            (AdversarialTest(real_conversation, simulated_conversation), ),
            return_exceptions=False
        ))[0]
        return cast(bool | None, result)
    
    @abstractmethod
    async def identify_real_conversations(
        self,
        instances: tuple[AdversarialTest, ...],
        return_exceptions: bool = True
    ) -> tuple[bool | None | Exception, ...]:
        """Evaluate multiple pairs of conversations concurrently. See identify_real_conversation for details.
        """
        ...

