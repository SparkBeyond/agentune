"""Zero-shot customer participant implementation."""

from .base import Customer
from ...models.conversation import Conversation
from ...models.message import Message


class ZeroShotCustomer(Customer):
    """Zero-shot LLM-based customer participant.
    
    Uses a language model to generate customer responses without
    fine-tuning or few-shot examples.
    """
    
    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate next customer message using zero-shot LLM approach.
        
        Args:
            conversation: Current conversation history
            
        Returns:
            Generated message or None if conversation should end
        """
        # TODO: Implement zero-shot customer logic
        # - Use LangChain LLM chain
        # - Apply customer persona/intent
        # - Generate realistic response timing
        raise NotImplementedError("ZeroShotCustomer implementation pending")
