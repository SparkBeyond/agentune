"""RAG-based customer participant implementation."""

from __future__ import annotations

import logging
from datetime import datetime
from attrs import frozen, field
from random import Random
import attrs
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from .first_message_prompt import CUSTOMER_FIRST_MESSAGE_PROMPT
from ....models import Conversation, Message
from ....rag import indexing_and_retrieval
from ..base import Customer, CustomerFactory
from .prompt import CUSTOMER_PROMPT
from ._customer_response import CustomerResponse


logger = logging.getLogger(__name__)


@frozen
class RagCustomer(Customer):
    """RAG LLM-based customer participant."""

    customer_vector_store: VectorStore
    model: BaseChatModel
    seed: int = 0
    intent_description: str | None = None
    _llm_chain: Runnable = field(init=False)
    _first_message_chain: Runnable = field(init=False)
    _random: Random = field(init=False)

    @_llm_chain.default
    def _create_llm_chain(self) -> Runnable:
        """Creates the LangChain Expression Language (LCEL) chain for the customer."""
        # Use the imported CUSTOMER_PROMPT from prompt.py
        prompt = CUSTOMER_PROMPT

        # Return the runnable chain with the imported prompt
        return prompt | self.model | PydanticOutputParser(pydantic_object=CustomerResponse)

    @_first_message_chain.default
    def _create_first_message_chain(self) -> Runnable:
        """Creates the chain for the first customer message."""
        # Use the imported CUSTOMER_PROMPT from prompt.py
        # This is a special case for the first message, which has a different prompt
        first_message_prompt = CUSTOMER_FIRST_MESSAGE_PROMPT

        # Return the runnable chain with the imported prompt
        return first_message_prompt | self.model | PydanticOutputParser(pydantic_object=CustomerResponse)

    @_random.default
    def _random_default(self) -> Random:
        return Random(self.seed)

    def with_intent(self, intent_description: str) -> RagCustomer:
        """Return a new RagCustomer instance with the specified intent."""
        return attrs.evolve(self, intent_description=intent_description)

    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate next customer message using RAG LLM approach."""
        # 1. Retrieval
        few_shot_examples: list[tuple[Document, float]] = await indexing_and_retrieval.get_few_shot_examples(
            conversation_history=conversation.messages,
            vector_store=self.customer_vector_store,
            k=20
        )

        # 2. Augmentation
        if not conversation.customer_messages:
            # If this is the first message by the customer, select one random example to provide context, to allow diverse options for the start of the conversation
            few_shot_examples = [self._random.choice(few_shot_examples)] if few_shot_examples else []
        else:
            # Select up to 5 randomly chosen examples
            few_shot_examples = self._random.sample(few_shot_examples, min(5, len(few_shot_examples)))

        # Format few-shot examples and history for the prompt template
        formatted_examples = indexing_and_retrieval.format_examples(few_shot_examples)

        # Format the current conversation in the same way as the examples
        formatted_current_convo = indexing_and_retrieval.format_conversation(conversation.messages)
        # Add the goal line to the conversation if there's an intent
        goal_line = (
            f"- Your goal in this conversation is: {self.intent_description}"
            if self.intent_description
            else ""
        )

        # 3. Generation
        if not conversation.customer_messages:
            # If this is the first message by the customer, use the first message chain
            chain = self._first_message_chain
        else:
            # Otherwise, use the regular chain
            chain = self._llm_chain

        chain_input = {
            "examples": formatted_examples,
            "current_conversation": formatted_current_convo,
            "goal_line": goal_line,
        }

        response_object: CustomerResponse = await chain.ainvoke(chain_input)

        if not response_object.should_respond or not response_object.response:
            return None

        # Use current timestamp for all messages
        response_timestamp = datetime.now()

        return Message(
            sender=self.role, content=response_object.response, timestamp=response_timestamp
        )


@frozen
class RagCustomerFactory(CustomerFactory):
    """Factory for creating RAG-based customer participants.

    Args:
        model: LangChain chat model for customer responses
        customer_vector_store: Vector store containing customer message examples
    """

    model: BaseChatModel
    customer_vector_store: VectorStore
    seed: int = 0
    _random: Random = field(init=False, repr=False)

    @_random.default
    def _create_random(self) -> Random:
        """Initialize random number generator with the specified seed."""
        return Random(self.seed)
    
    def create_participant(self) -> RagCustomer:
        """Create a RAG customer participant.
        
        Returns:
            RagCustomer instance configured with the vector store
        """
        return RagCustomer(
            customer_vector_store=self.customer_vector_store,
            model=self.model,
            seed=self._random.randint(0, 1000)
        )
