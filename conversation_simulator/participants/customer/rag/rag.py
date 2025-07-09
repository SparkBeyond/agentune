"""RAG-based customer participant implementation."""

from __future__ import annotations

import logging
from datetime import datetime
from random import Random

from attrs import frozen, field
import attrs
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from ....models import Conversation, Message
from ....rag import indexing_and_retrieval
from ..base import Customer, CustomerFactory
from .prompt import CUSTOMER_PROMPT, CustomerResponse

logger = logging.getLogger(__name__)

@frozen
class RagCustomer(Customer):
    """RAG LLM-based customer participant."""

    customer_vector_store: VectorStore
    model: BaseChatModel
    seed: int = 0
    intent_description: str | None = None
    llm_chain: Runnable = field(init=False)
    _random: Random = field(init=False, repr=False)

    @llm_chain.default
    def _create_llm_chain(self) -> Runnable:
        """Creates the LangChain Expression Language (LCEL) chain for the customer."""
        # Use the imported CUSTOMER_PROMPT from prompt.py
        prompt = CUSTOMER_PROMPT
        
        # Return the runnable chain with the imported prompt
        return prompt | self.model | PydanticOutputParser(pydantic_object=CustomerResponse)
    
    @_random.default
    def _create_random(self) -> Random:
        """Create a random number generator with the specified seed."""
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

        # 2. Calculate probability of returning a message for the target role
        probability = await indexing_and_retrieval.probability_of_next_message_for(
            role=self.role,
            similar_docs=few_shot_examples
        )

        if not conversation.customer_messages:
            probability_description = ""
        else:
            probability_description = f"The probability that the customer would respond at this point (based on similar conversation patterns in the historical data) is estimated at: {probability:.2f}"

        # 3. Examples selection and formatting
        if not conversation.customer_messages:
            # If this is the first message by the customer, select one random example to provide context, to allow diverse options for the start of the conversation
            few_shot_examples = [self._random.choice(few_shot_examples)]
        else:
            # Select up to 5 randomly chosen examples
            few_shot_examples = self._random.sample(few_shot_examples, min(5, len(few_shot_examples)))

        # Format few-shot examples and history for the prompt template
        formatted_examples = self._format_examples(few_shot_examples)
        formatted_current_conversation = indexing_and_retrieval._format_conversation_history(conversation.messages)

        # 4. Intent statement
        # Add the goal line to the conversation if there's an intent
        if self.intent_description:
            goal_line = f"- Your goal in this conversation is: {self.intent_description}"
        else:
            goal_line = ""

        # 5. Chain execution
        chain_input = {
            "examples": formatted_examples,
            "current_conversation": formatted_current_conversation,
            "probability_description": probability_description,
            "goal_line": goal_line
        }
        response: CustomerResponse = await self.llm_chain.ainvoke(chain_input)

        log_message = f"{self.role.value} - retrieved {len(few_shot_examples)} examples, probability : {probability}, decided to respond: {response.should_respond}"
        if response.should_respond:
            log_message = log_message + f"\n==> {response.response}"
        logger.debug(log_message)

        # 6. Process response
        # Check if the customer should respond
        if not response.should_respond:
            return None
        
        if response.response is None:
            logger.error("Customer response is None, even though should_respond is True!")
            return None

        # Use current timestamp for all messages
        response_timestamp = datetime.now()

        return Message(
            sender=self.role,
            content=response.response,  # Now guaranteed to be not None
            timestamp=response_timestamp
        )

    def _format_examples(self, examples: list[tuple[Document, float]]) -> str:
        """
        Formats the retrieved few-shot example Documents into a string format.
        Each Document's metadata contains the full conversation example.
        """
        def _format_example(doc: Document, num: int) -> str:
            return f"Example conversation {num}:\n{doc.metadata['full_conversation']}"

        conversations = [
            _format_example(doc, i) for i, (doc, _) in enumerate(examples)
        ]

        return "\n\n".join(conversations)


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
        """Create a random number generator with the specified seed."""
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
