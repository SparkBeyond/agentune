"""RAG-based agent participant implementation."""

from __future__ import annotations
import logging # Added
import random
from datetime import datetime, timedelta
from typing import List

from langchain_core.documents import Document # Added
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage # AIMessage, HumanMessage Added
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore # Added VectorStore for type hint consistency
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ....models import Conversation, Message, ParticipantRole
from ..base import Agent

from ....rag import get_few_shot_examples_for_agent

logger = logging.getLogger(__name__)


class RagAgent(Agent):
    """RAG LLM-based agent participant.
    
    Uses a language model with Retrieval Augmented Generation
    to generate agent responses.
    """

    def __init__(
        self,
        agent_vector_store: VectorStore,
        openai_api_key: str,
    ):
        """Initializes the RAG agent.

        Args:
            agent_vector_store: Vector store for agent messages.
            openai_api_key: The OpenAI API key.
        """
        super().__init__()
        self.agent_vector_store = agent_vector_store
        self.openai_api_key = openai_api_key
        self.intent_description: str | None = None # Store intent
        self.llm_chain = self._create_llm_chain()
    
    def _create_llm_chain(self, intent_description: str | None = None) -> Runnable:
        """Creates the LangChain Expression Language (LCEL) chain for the agent."""
        # Construct the system prompt
        base_system_prompt = """You are a helpful customer service agent.
- Your primary goal is to assist the user with their issue based on the conversation history and relevant examples.
- If few-shot examples are provided, use them to understand the tone, style, and common solutions.
- If no examples are relevant, rely on your general knowledge and the conversation history.
- Keep your responses clear, concise, and professional."""

        current_intent = intent_description or self.intent_description
        if current_intent:
            system_prompt_content = f"Your primary goal is: {current_intent}\n\n{base_system_prompt}"
        else:
            system_prompt_content = base_system_prompt

        # The `MessagesPlaceholder` is a special variable that can hold a sequence of messages.
        # `few_shot_examples` will be a list of alternating Human/AI messages.
        # `chat_history` will be the messages from the current conversation.
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt_content),
            MessagesPlaceholder(variable_name="few_shot_examples"),
            MessagesPlaceholder(variable_name="chat_history"),
        ])

        llm = ChatOpenAI(api_key=SecretStr(self.openai_api_key), model="gpt-4o-mini")

        # Return the runnable chain
        return prompt_template | llm | StrOutputParser()

    def with_intent(self, intent_description: str) -> RagAgent:
        """Return a new RagAgent instance with the specified intent.
        
        The intent is primarily used to guide the initial system prompt.
        """
        new_agent = RagAgent(
            agent_vector_store=self.agent_vector_store,
            openai_api_key=self.openai_api_key,
        )
        new_agent.intent_description = intent_description
        # Re-create the LLM chain with the new intent description
        new_agent.llm_chain = new_agent._create_llm_chain(intent_description=intent_description)
        return new_agent

    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate next agent message using RAG LLM approach."""
        if not conversation.messages or conversation.messages[-1].sender != ParticipantRole.CUSTOMER:
            return None

        # 1. Retrieval
        few_shot_examples: List[Message] = await get_few_shot_examples_for_agent(
            conversation_history=list(conversation.messages),
            agent_vector_store=self.agent_vector_store,
            k=3,
        )

        # 2. Augmentation
        # Format few-shot examples and history for the prompt template
        formatted_examples = self._format_examples(few_shot_examples)
        chat_history = conversation.to_langchain_messages()

        # 3. Generation
        response_content = await self.llm_chain.ainvoke({
            "few_shot_examples": formatted_examples,
            "chat_history": chat_history,
        })

        if not response_content.strip():
            return None

        if conversation.messages:
            last_timestamp = conversation.messages[-1].timestamp
            delay_seconds = random.randint(1, 5)  # Agents respond quickly
            response_timestamp = last_timestamp + timedelta(seconds=delay_seconds)
        else:
            # This case should ideally not be hit if conversations always have a start
            response_timestamp = datetime.now()

        return Message(
            sender=self.role,
            content=response_content,
            timestamp=response_timestamp,
        )

    def _format_examples(self, examples: List[Document]) -> List[BaseMessage]:
        """
        Formats the retrieved few-shot example Documents into a list of LangChain messages.
        Each Document's page_content (history, typically customer's turn) becomes a HumanMessage,
        and the metadata (next agent message) becomes an AIMessage.
        """
        formatted_messages: List[BaseMessage] = []
        for doc in examples:
            try:
                # History (customer's turn or context)
                history_content = doc.page_content
                if not history_content.strip(): # Ensure history is not empty
                    logger.warning("Skipping few-shot example with empty history (page_content) in RagAgent.")
                    continue
                formatted_messages.append(HumanMessage(content=history_content))

                # Example agent response from metadata
                metadata = doc.metadata
                # Validate essential keys before creating Message object
                if not all(key in metadata for key in ["role", "content", "timestamp"]):
                    logger.warning(f"Skipping few-shot example due to missing essential metadata: {metadata} in RagAgent.")
                    continue
                
                agent_message = Message(
                    sender=ParticipantRole(metadata["role"]),
                    content=str(metadata["content"]),
                    timestamp=datetime.fromisoformat(str(metadata["timestamp"])),
                )

                if agent_message.sender != ParticipantRole.AGENT:
                    logger.warning(
                        f"Skipping example with unexpected role '{agent_message.sender}' in RagAgent._format_examples. Expected AGENT."
                    )
                    continue
                
                # Ensure content is not empty before adding
                if not agent_message.content.strip():
                    logger.warning("Skipping few-shot example with empty content for AIMessage in RagAgent.")
                    continue

                formatted_messages.append(agent_message.to_langchain())  # This will be AIMessage
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Error processing document for few-shot example in RagAgent: {doc}. Error: {e}. Skipping.")
                continue
        return formatted_messages
