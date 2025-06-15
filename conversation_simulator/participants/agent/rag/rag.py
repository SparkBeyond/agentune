"""RAG-based agent participant implementation."""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from ....models import Conversation, Message, ParticipantRole
from ....rag import _get_few_shot_examples
from ..base import Agent

logger = logging.getLogger(__name__)


class RagAgent(Agent):
    """RAG LLM-based agent participant.
    
    Uses a language model with Retrieval Augmented Generation
    to generate agent responses.
    """

    def __init__(
        self,
        agent_vector_store: VectorStore,
        model: BaseChatModel
    ):
        """Initializes the RAG agent.

        Args:
            agent_vector_store: Vector store containing agent messages.
            model: The LLM model name to use.
        """
        super().__init__()
        self.agent_vector_store = agent_vector_store
        self.model = model
        self.intent_description: str | None = None # Store intent
        self.llm_chain = self._create_llm_chain(model=model)
    
    def _create_llm_chain(self, model: BaseChatModel) -> Runnable:
        """Creates the LangChain Expression Language (LCEL) chain for the agent."""
        # Construct the system prompt
        base_system_prompt = """You are a helpful customer service agent.
- Your primary goal is to assist the user with their issue based on the conversation history and relevant examples.
- If few-shot examples are provided, use them to understand the tone, style, and common solutions.
- If no examples are relevant, rely on your general knowledge and the conversation history.
- Keep your responses clear, concise, and professional."""

        if self.intent_description:
            system_prompt_content = f"Your primary goal is: {self.intent_description}\n\n{base_system_prompt}"
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

        # Return the runnable chain
        return prompt_template | model | StrOutputParser()

    def with_intent(self, intent_description: str) -> RagAgent:
        """Return a new RagAgent instance with the specified intent."""
        new_agent = RagAgent(
            agent_vector_store=self.agent_vector_store,
            model=self.model
        )
        new_agent.intent_description = intent_description
        return new_agent

    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate next agent message using RAG LLM approach."""
        if not conversation.messages or conversation.messages[-1].sender != ParticipantRole.CUSTOMER:
            return None

        # 1. Retrieval
        few_shot_examples: List[Document] = await self.get_few_shot_examples_for_agent(
            conversation.messages,
            k=3,
            vector_store=self.agent_vector_store
        )

        # 2. Augmentation
        # Format few-shot examples and history for the prompt template
        formatted_examples = self._format_examples(few_shot_examples)
        chat_history: List[BaseMessage] = conversation.to_langchain_messages()

        # 3. Generation
        chain = self._create_llm_chain(model=self.model)
        response_content = await chain.ainvoke({
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

    @staticmethod
    async def get_few_shot_examples_for_agent(conversation_history: Sequence[Message], vector_store: VectorStore, k: int = 3) -> List[Document]:
        return await _get_few_shot_examples(
            conversation_history=conversation_history,
            vector_store=vector_store,
            k=k,
            target_role=ParticipantRole.AGENT
        )