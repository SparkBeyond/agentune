"""RAG-based customer participant implementation."""

from __future__ import annotations
import logging # Added
import random
from datetime import datetime, timedelta
from typing import List

from langchain_core.documents import Document # Added
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage # AIMessage, HumanMessage Added
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore # Added VectorStore for type hint consistency
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ....models import Conversation, Message, ParticipantRole
from ..base import Customer
from ....rag import get_few_shot_examples_for_customer

logger = logging.getLogger(__name__)


CUSTOMER_SYSTEM_PROMPT = """You are a customer interacting with a customer service agent.
- Your goal is to resolve your issue.
- Behave like a real person. You can be frustrated, confused, or happy depending on the agent's responses.
- Use the provided few-shot examples to understand the tone and style of a typical customer in this situation.
- Use the conversation history to stay in context.
- Keep your responses concise and to the point.
"""


class RagCustomer(Customer):
    """RAG LLM-based customer participant."""

    llm_chain: Runnable

    def __init__(
        self,
        customer_vector_store: VectorStore,
        openai_api_key: str,
    ):
        super().__init__()
        self.customer_vector_store = customer_vector_store
        self.openai_api_key = openai_api_key
        self.intent_description: str | None = None
        self.llm_chain = self._create_llm_chain()

    def _create_llm_chain(self, intent_description: str | None = None) -> Runnable:
        """Creates the LangChain Expression Language (LCEL) chain for the customer."""
        base_system_prompt = CUSTOMER_SYSTEM_PROMPT

        if intent_description:
            system_prompt_content = f"Your goal: {intent_description}\n\n{base_system_prompt}"
        elif self.intent_description:
            system_prompt_content = f"Your goal: {self.intent_description}\n\n{base_system_prompt}"
        else:
            system_prompt_content = base_system_prompt

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt_content),
                MessagesPlaceholder(variable_name="few_shot_examples"),
                MessagesPlaceholder(variable_name="chat_history"),
            ]
        )
        llm = ChatOpenAI(api_key=SecretStr(self.openai_api_key), model="gpt-4o-mini")
        return prompt | llm | StrOutputParser()

    def with_intent(self, intent_description: str) -> RagCustomer:
        """Return a new RagCustomer instance with the specified intent."""
        new_customer = RagCustomer(
            customer_vector_store=self.customer_vector_store,
            openai_api_key=self.openai_api_key,
        )
        new_customer.intent_description = intent_description
        new_customer.llm_chain = new_customer._create_llm_chain(intent_description=intent_description)
        return new_customer

    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate next customer message using RAG LLM approach."""
        if not conversation.messages or conversation.messages[-1].sender != ParticipantRole.AGENT:
            return None

        # 1. Retrieval
        few_shot_examples: List[Message] = await get_few_shot_examples_for_customer(
            conversation_history=list(conversation.messages),
            customer_vector_store=self.customer_vector_store,
        )

        # 2. Augmentation
        formatted_examples = self._format_examples(few_shot_examples)
        chat_history_langchain = conversation.to_langchain_messages()

        # 3. Generation
        response_content = await self.llm_chain.ainvoke(
            {
                "few_shot_examples": formatted_examples,
                "chat_history": chat_history_langchain,
            }
        )

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
                    content=str(metadata["content"]),
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

                formatted_messages.append(customer_message.to_langchain())  # This will be HumanMessage
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Error processing document for few-shot example in RagCustomer: {doc}. Error: {e}. Skipping.")
                continue
        return formatted_messages
