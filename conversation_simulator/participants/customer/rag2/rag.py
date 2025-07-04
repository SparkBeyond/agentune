"""RAG-based customer participant implementation."""

from __future__ import annotations

import logging
import random
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field

from ....models import Conversation, Message
from ....rag import index_by_prefix
from ..base import Customer, CustomerFactory
from .prompt import CUSTOMER_PROMPT

logger = logging.getLogger(__name__)


class CustomerResponse(BaseModel):
    """Customer's response (if relevant) with reasoning."""
    
    reasoning: str = Field(
        description="Detailed reasoning for why the customer would respond or not, and what the response would be"
    )
    should_respond: bool = Field(
        description="Whether the customer should respond at this point"
    )
    response: str | None = Field(
        default=None,
        description="Response content, or null if should_respond is false"
    )

class RagCustomer(Customer):
    """RAG LLM-based customer participant.
    
    Uses a language model with Retrieval Augmented Generation
    to generate customer responses.
    """

    def __init__(
        self,
        customer_vector_store: VectorStore,
        model: BaseChatModel,
        seed: int = 0
    ):
        """Initializes the RAG customer.

        Args:
            customer_vector_store: Vector store containing customer messages.
            model: The LLM model name to use.
            seed: Random seed for deterministic behavior.
        """
        super().__init__()
        self.customer_vector_store = customer_vector_store
        self.model = model
        self.intent_description: str | None = None # Store intent
        self._output_parser = PydanticOutputParser(pydantic_object=CustomerResponse)
        self.llm_chain = self._create_llm_chain(model=model)
        self._random = random.Random(seed)

    def _create_llm_chain(self, model: BaseChatModel) -> Runnable:
        """Creates the LangChain Expression Language (LCEL) chain for the customer."""
        prompt = CUSTOMER_PROMPT
        
        # Return the runnable chain with the imported prompt
        return prompt | model | self._output_parser

    def with_intent(self, intent_description: str) -> RagCustomer:
        """Return a new RagCustomer instance with the specified intent."""
        new_customer = RagCustomer(
            customer_vector_store=self.customer_vector_store,
            model=self.model
        )
        new_customer.intent_description = intent_description
        return new_customer

    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate next customer message using RAG LLM approach."""
        # if not conversation.messages or conversation.messages[-1].sender != ParticipantRole.AGENT:
        #     return None

        # 1. Retrieval
        few_shot_examples: list[tuple[Document, float]] = await index_by_prefix.get_few_shot_examples(
            conversation_history=conversation.messages,
            vector_store=self.customer_vector_store,
            k=20
        )
        logger.debug(f"Role: {self.role}, Retrieved {len(few_shot_examples)} few-shot examples")

        # 2. Calculate probability of returning a message for the target role
        probability = await index_by_prefix.probability_of_next_message_for(
            role=self.role,
            similar_docs=few_shot_examples
        )
        logger.debug(f"Role: {self.role}, Probability of next message: {probability}")
        
        # Use current timestamp instead of predicting one
        current_timestamp = datetime.now()

        # 3. Augmentation
        # Select few-shot examples randomly, to avoid bias
        few_shot_examples = self._random.sample(few_shot_examples, min(5, len(few_shot_examples)))

        # Format few-shot examples and history for the prompt template
        formatted_examples = self._format_examples(few_shot_examples)
        formatted_current_conversation = index_by_prefix._format_conversation_history(conversation.messages)

        # Add the goal line to the conversation if there's an intent
        if self.intent_description:
            goal_line = f"This conversation was initiated by the customer with the following intent:\n{self.intent_description}"
        else:
            goal_line = ""
        
        # 4. Generation
        chain = self._create_llm_chain(model=self.model)
        response: CustomerResponse = await chain.ainvoke({
            "examples": formatted_examples,
            "current_conversation": formatted_current_conversation,
            "goal_line": goal_line,
            "format_instructions": self._output_parser.get_format_instructions()
        })

        if not response.should_respond:
            logger.debug(f"Role: {self.role}, LLM decided not to respond: {response.reasoning}")
            return None

        if response.response is None:
            logger.error("Customer response is None, even though should_respond is True!")
            return None

        logger.debug(f"Role: {self.role}, LLM decided to respond: {response.reasoning}. Response: {response.response}")
        
        return Message(
            sender=self.role,
            content=response.response,  # Now guaranteed to be not None
            timestamp=current_timestamp
        )

    def _format_examples(self, examples: list[tuple[Document, float]]) -> str:
        """
        Formats the retrieved few-shot example
        """
        conversations = [
            f"Example conversation {i+1}:\n{doc.metadata['full_conversation']}" 
            for i, (doc, _) in enumerate(examples)
        ]

        return "\n\n".join(conversations)


class RagCustomerFactory(CustomerFactory):
    """Factory for creating RAG-based customer participants."""
    
    def __init__(
        self, 
        model: BaseChatModel, 
        customer_vector_store: VectorStore,
        seed: int = 0
    ) -> None:
        """Initialize the factory.
        
        Args:
            model: LangChain chat model for customer responses
            customer_vector_store: Vector store containing customer message examples
            seed: Random seed for deterministic behavior
        """
        self.model = model
        self.customer_vector_store = customer_vector_store
        self.seed = seed
    
    def create_participant(self) -> RagCustomer:
        """Create a RAG customer participant.
        
        Returns:
            RagCustomer instance configured with the vector store
        """
        return RagCustomer(
            customer_vector_store=self.customer_vector_store,
            model=self.model,
            seed=self.seed
        )
