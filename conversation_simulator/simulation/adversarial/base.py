"""Base class for adversarial testing of conversation simulation quality.

Adversarial testing evaluates how well simulated conversations can be distinguished
from real conversations by testing whether a model or human can identify which
conversation is real when presented with a pair.
"""

import random
from abc import ABC, abstractmethod

from ...models.conversation import Conversation

# Set random seed for reproducible adversarial testing
random.seed(42)


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
        conversation_1: Conversation,
        conversation_2: Conversation,
    ) -> int:
        """Identify which of two conversations is the real one.
        
        Given a pair of conversations where one is real and one is simulated,
        determine which one appears to be the authentic human conversation.
        
        Args:
            conversation_1: First conversation to evaluate
            conversation_2: Second conversation to evaluate
            
        Returns:
            Index of the conversation believed to be real (0 or 1)
            - 0 indicates conversation_1 is believed to be real
            - 1 indicates conversation_2 is believed to be real
            
        Note:
            The caller is responsible for ensuring exactly one conversation
            is real and one is simulated. The tester should not assume
            any particular ordering.
        """
        pass
    
    async def evaluate_pair(
        self,
        real_conversation: Conversation,
        simulated_conversation: Conversation,
    ) -> bool:
        """Evaluate a specific real/simulated pair and return correctness.
        
        This is a convenience method that handles the randomization of
        conversation order and evaluates whether the tester correctly
        identified the real conversation.
        
        Args:
            real_conversation: The authentic human conversation
            simulated_conversation: The AI-generated conversation
            
        Returns:
            True if the tester correctly identified the real conversation,
            False otherwise
        """
        # Randomly order the conversations
        if random.choice([True, False]):
            # Real conversation is first
            conversations = [real_conversation, simulated_conversation]
            real_index = 0
        else:
            # Simulated conversation is first
            conversations = [simulated_conversation, real_conversation]
            real_index = 1
        
        # Get the tester's prediction
        predicted_real_index = await self.identify_real_conversation(
            conversations[0], conversations[1]
        )
        
        # Return whether the prediction was correct
        return predicted_real_index == real_index
