"""RAG-based customer participant implementation."""

from __future__ import annotations

import logging
from datetime import datetime
from collections.abc import Sequence

from attrs import frozen
import attrs
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from ....models import Conversation, Message, ParticipantRole
from ....rag import indexing_and_retrieval
from ..base import Customer, CustomerFactory
from .prompt import CUSTOMER_PROMPT


class CustomerResponse(BaseModel):
    """Customer's response with reasoning."""

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

logger = logging.getLogger(__name__)

@frozen
class RagCustomer(Customer):
    """RAG LLM-based customer participant."""

    customer_vector_store: VectorStore
    model: BaseChatModel
    intent_description: str | None = None

    def _create_llm_chain(self, model: BaseChatModel) -> Runnable:
        """Creates the LangChain Expression Language (LCEL) chain for the customer."""
        # Use the imported CUSTOMER_PROMPT from prompt.py
        # The customer goal will be passed in the invoke parameters
        prompt = CUSTOMER_PROMPT
        
        # Prepare the chain with the prompt, model, and output parsing
        return prompt | model | PydanticOutputParser(pydantic_object=CustomerResponse)

    def with_intent(self, intent_description: str) -> RagCustomer:
        """Return a new RagCustomer instance with the specified intent."""
        return attrs.evolve(self, intent_description=intent_description)

    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate next customer message using RAG LLM approach."""
        #if not conversation.messages or conversation.messages[-1].sender != ParticipantRole.AGENT:
        #    return None

        # 1. Retrieval
        few_shot_examples: list[Document] = await self._get_few_shot_examples(
            conversation.messages, k=3, vector_store=self.customer_vector_store
        )

        # 2. Augmentation
        formatted_examples = self._format_examples(few_shot_examples)

        # 3. Generation
        chain = self._create_llm_chain(model=self.model)

        # Add the goal line to the conversation if there's an intent
        goal_line = (
            f"- Your goal in this conversation is: {self.intent_description}"
            if self.intent_description
            else ""
        )

        # Prepare the input for the chain
        # Format the current conversation in the same way as the examples
        formatted_current_convo = "\n".join(
            [f"{msg.sender.value.capitalize()}: {msg.content}" for msg in conversation.messages]
        )

        chain_input = {
            "examples": "\n\n".join([str(msg.content) for msg in formatted_examples]),
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

    def _format_examples(
        self, examples: list[Document]
    ) -> list[BaseMessage]:
        """
        Formats the retrieved few-shot example Documents into a list of LangChain messages.
        Each Document's page_content (history, typically agent's turn) becomes an AIMessage,
        and the metadata (next customer message) becomes an embedded customer response in the conversation history.
        """
        formatted_messages: list[BaseMessage] = []
        for index, doc in enumerate(examples):
            try:
                # Example customer response from metadata
                metadata = doc.metadata
                # Validate essential keys
                if not all(
                    key in metadata
                    for key in ["next_message_content", "full_conversation"]
                ):
                    logger.warning(f"Skipping few-shot example due to missing essential metadata: {metadata} in RagCustomer.")
                    continue

                full_conversation = str(metadata["full_conversation"])
                customer_response = str(metadata["next_message_content"])

                # Replace the customer's turn in the conversation with the desired format
                customer_turn_str = f"Customer: {customer_response}"
                if customer_turn_str in full_conversation:
                    modified_conversation = full_conversation.replace(
                        customer_turn_str, f"**Current customer response**: {customer_response}"
                    )
                else:
                    logger.warning(
                        f"Could not find customer turn '{customer_turn_str}' in full conversation. Using original."
                    )
                    modified_conversation = full_conversation

                # Add indexing for the examples
                # We use AIMessage here because the history is from the Agent's perspective
                formatted_messages.append(AIMessage(content=f"Example {index + 1}:"))
                formatted_messages.append(AIMessage(content=modified_conversation))

            except Exception as e:
                logger.error(f"Error processing few-shot example in RagCustomer: {e}")
                continue
        return formatted_messages

    @staticmethod
    async def _get_few_shot_examples(conversation_history: Sequence[Message], vector_store: VectorStore, k: int = 3) -> list[Document]:
        return await indexing_and_retrieval.get_similar_examples_for_next_message_role(
            conversation_history=conversation_history,
            vector_store=vector_store,
            k=k,
            target_role=ParticipantRole.CUSTOMER
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
    
    def create_participant(self) -> RagCustomer:
        """Create a RAG customer participant.
        
        Returns:
            RagCustomer instance configured with the vector store
        """
        return RagCustomer(
            customer_vector_store=self.customer_vector_store,
            model=self.model
        )
