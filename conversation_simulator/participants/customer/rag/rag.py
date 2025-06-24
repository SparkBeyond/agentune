"""RAG-based customer participant implementation."""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from ....models import Conversation, Message, ParticipantRole
from ....rag import get_few_shot_examples
from ..base import Customer, CustomerFactory
from .prompt import CUSTOMER_PROMPT

logger = logging.getLogger(__name__)

class RagCustomer(Customer):
    """RAG LLM-based customer participant."""

    def __init__(
        self,
        customer_vector_store: VectorStore,
        model: BaseChatModel
    ):
        super().__init__()
        self.customer_vector_store = customer_vector_store
        self.model = model
        self.intent_description: str | None = None

    def _create_llm_chain(self, model: BaseChatModel) -> Runnable:
        """Creates the LangChain Expression Language (LCEL) chain for the customer."""
        # Use the imported CUSTOMER_PROMPT from prompt.py
        # The customer goal will be passed in the invoke parameters
        prompt = CUSTOMER_PROMPT
        
        # Prepare the chain with the imported prompt
        return prompt | model | StrOutputParser()

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
        if not conversation.messages or conversation.messages[-1].sender != ParticipantRole.AGENT:
            return None

        # 1. Retrieval
        few_shot_examples: List[Document] = await self._get_few_shot_examples(
            conversation.messages,
            k=3,
            vector_store=self.customer_vector_store
        )

        # 2. Augmentation
        formatted_examples = self._format_examples(few_shot_examples)
        chat_history_langchain: List[BaseMessage] = conversation.to_langchain_messages()

        # 3. Generation
        chain = self._create_llm_chain(model=self.model)

        
        # Add the goal line to the conversation if there's an intent
        goal_line = f"- Your goal in this conversation is: {self.intent_description}" if self.intent_description else ""
        
        # Prepare the input for the chain
        chain_input = {
            "examples": "\n\n".join([str(msg.content) for msg in formatted_examples]),
            "current_conversation": "\n".join([str(msg.content) for msg in chat_history_langchain]),
            "goal_line": goal_line
        }
            
        response_content = await chain.ainvoke(chain_input)

        if not response_content.strip():
            return None

        if conversation.messages:
            last_timestamp = conversation.messages[-1].timestamp
            delay_seconds = random.randint(5, 30)  # Customers take longer
            response_timestamp = last_timestamp + timedelta(seconds=delay_seconds)
        else:
            response_timestamp = datetime.now()

        return Message(
            sender=self.role, content=response_content, timestamp=response_timestamp
        )

    def _format_examples(
        self, examples: List[Document]
    ) -> List[BaseMessage]:
        """
        Formats the retrieved few-shot example Documents into a list of LangChain messages.
        Each Document's page_content (history, typically agent's turn) becomes an AIMessage,
        and the metadata (next customer message) becomes a HumanMessage.
        """
        formatted_messages: List[BaseMessage] = []
        for doc in examples:
            try:
                # History (agent's turn or context)
                history_content = doc.page_content
                if not history_content.strip(): # Ensure history is not empty
                    logger.warning("Skipping few-shot example with empty history (page_content) in RagCustomer.")
                    continue
                formatted_messages.append(AIMessage(content=history_content))

                # Example customer response from metadata
                metadata = doc.metadata
                # Validate essential keys before creating Message object
                if not all(key in metadata for key in ["role", "content", "timestamp"]):
                    logger.warning(f"Skipping few-shot example due to missing essential metadata: {metadata} in RagCustomer.")
                    continue
                
                customer_message = Message(
                    sender=ParticipantRole(metadata["role"]),
                    content='Customer response: ' + str(metadata["content"]),
                    timestamp=datetime.fromisoformat(str(metadata["timestamp"])),
                )

                if customer_message.sender != ParticipantRole.CUSTOMER:
                    logger.warning(
                        f"Skipping example with unexpected role '{customer_message.sender}' in RagCustomer._format_examples. Expected CUSTOMER."
                    )
                    continue
                
                # Ensure content is not empty before adding
                if not customer_message.content.strip():
                    logger.warning("Skipping few-shot example with empty content for HumanMessage in RagCustomer.")
                    continue

                customer_message_langchain = customer_message.to_langchain()
                customer_message_langchain.content = 'Customer response: ' + str(customer_message.content)
                formatted_messages.append(customer_message_langchain)

            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Error processing document for few-shot example in RagCustomer: {doc}. Error: {e}. Skipping.")
                continue
        return formatted_messages

    @staticmethod
    async def _get_few_shot_examples(conversation_history: Sequence[Message], vector_store: VectorStore, k: int = 3) -> List[Document]:
        return await get_few_shot_examples(
            conversation_history=conversation_history,
            vector_store=vector_store,
            k=k,
            target_role=ParticipantRole.CUSTOMER
        )


class RagCustomerFactory(CustomerFactory):
    """Factory for creating RAG-based customer participants."""
    
    def __init__(
        self, 
        model: BaseChatModel, 
        customer_vector_store: VectorStore
    ) -> None:
        """Initialize the factory.
        
        Args:
            model: LangChain chat model for customer responses
            customer_vector_store: Vector store containing customer message examples
        """
        self.model = model
        self.customer_vector_store = customer_vector_store
    
    def create_participant(self) -> RagCustomer:
        """Create a RAG customer participant.
        
        Returns:
            RagCustomer instance configured with the vector store
        """
        return RagCustomer(
            customer_vector_store=self.customer_vector_store,
            model=self.model
        )
