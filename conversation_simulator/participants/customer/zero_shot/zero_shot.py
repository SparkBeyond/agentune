"""Zero-shot customer participant implementation."""

from datetime import datetime

import attrs
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from ....models.conversation import Conversation
from ....models.message import Message
from ..base import Customer, CustomerFactory
from .prompts import CustomerPromptBuilder


@attrs.frozen
class ZeroShotCustomer(Customer):
    """Zero-shot LLM-based customer participant.
    
    Uses a language model to generate customer responses without
    fine-tuning or few-shot examples.
    
    This class is immutable - use with_intent() to create new instances
    with different intents.
    """
    
    model: BaseChatModel
    intent_description: str | None = None
    
    # Private fields for internal state
    _prompt_builder: CustomerPromptBuilder = attrs.field(factory=CustomerPromptBuilder, init=False)
    _output_parser: StrOutputParser = attrs.field(factory=StrOutputParser, init=False)
    
    def with_intent(self, intent_description: str) -> 'ZeroShotCustomer':
        """Return a new customer instance with the specified intent.
        
        Args:
            intent_description: Natural language description of the customer's goal/intent
            
        Returns:
            New ZeroShotCustomer instance with the intent installed
        """
        return attrs.evolve(self, intent_description=intent_description)
    
    def _get_chain(self) -> Runnable:
        """Create the LangChain processing chain."""
        prompt_template = self._prompt_builder.build_chat_template(
            intent_description=self.intent_description
        )
        return prompt_template | self.model | self._output_parser
    
    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate next customer message using zero-shot LLM approach.
        
        Args:
            conversation: Current conversation history
            
        Returns:
            Generated message or None if conversation should end
        """
        # Convert conversation to messages for the chain
        conversation_history = conversation.to_langchain_messages()
        
        # Use the chain to get response
        chain = self._get_chain()
        customer_response = await chain.ainvoke({
            "conversation_history": conversation_history
        })
        
        # If message is empty, do not respond
        if not customer_response.strip():
            return None
        else:
            # Use current timestamp for all messages
            response_timestamp = datetime.now()
            
            return Message(
                sender=self.role,
                content=customer_response.strip(),
                timestamp=response_timestamp
            )

@attrs.frozen
class ZeroShotCustomerFactory(CustomerFactory):
    """Factory for creating zero-shot customer participants."""
    
    model: BaseChatModel
    
    def create_participant(self) -> ZeroShotCustomer:
        """Create a zero-shot customer participant.
        
        Returns:
            ZeroShotCustomer instance
        """
        return ZeroShotCustomer(model=self.model)
