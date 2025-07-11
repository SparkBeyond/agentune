"""RAG-based agent participant implementation."""

from __future__ import annotations

import logging
from datetime import datetime
from collections.abc import Sequence

from attrs import field, frozen
import attrs
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from ....models import Conversation, Message, ParticipantRole
from ....rag import indexing_and_retrieval
from ..base import Agent, AgentFactory
from ..config import AgentConfig
from .prompt import AGENT_PROMPT

logger = logging.getLogger(__name__)


class AgentResponse(BaseModel):
    """Agent's response with reasoning."""

    reasoning: str = Field(
        description="Detailed reasoning for why the agent would respond or not, and what the response would be"
    )
    should_respond: bool = Field(
        description="Whether the agent should respond at this point"
    )
    response: str | None = Field(
        default=None,
        description="Response content, or null if should_respond is false"
    )

@frozen
class RagAgent(Agent):
    """RAG LLM-based agent participant.
    
    Uses a language model with Retrieval Augmented Generation
    to generate agent responses.
    """

    agent_vector_store: VectorStore
    model: BaseChatModel
    intent_description: str | None = None # Store intent
        
    llm_chain: Runnable = field(init=False)
    
    @llm_chain.default
    def _create_llm_chain(self) -> Runnable:
        """Creates the LangChain Expression Language (LCEL) chain for the agent."""
        # Use the imported AGENT_PROMPT from prompt.py
        # If there's an intent description, we can modify the system message
        prompt = AGENT_PROMPT
        
        # Return the runnable chain with the imported prompt
        return prompt | self.model | PydanticOutputParser(pydantic_object=AgentResponse)

    def with_intent(self, intent_description: str) -> RagAgent:
        """Return a new RagAgent instance with the specified intent."""
        return attrs.evolve(self, intent_description=intent_description)

    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate next agent message using RAG LLM approach."""
        #if not conversation.messages or conversation.messages[-1].sender != ParticipantRole.CUSTOMER:
        #    return None

        # 1. Retrieval
        few_shot_examples: list[Document] = await self._get_few_shot_examples(
            conversation.messages, k=3, vector_store=self.agent_vector_store
        )

        # 2. Augmentation
        # Format few-shot examples and history for the prompt template
        formatted_examples = self._format_examples(few_shot_examples)

        # 3. Generation
        formatted_current_convo = "\n".join(
            [f"{msg.sender.value.capitalize()}: {msg.content}" for msg in conversation.messages]
        )
        response_content: AgentResponse = await self.llm_chain.ainvoke({
            "examples": "\n\n".join([str(msg.content) for msg in formatted_examples]),
            "current_conversation": formatted_current_convo,
        })

        if not response_content.should_respond or not response_content.response:
            return None

        # Guardrail: Check for repeated messages
        if conversation.messages:
            last_message = conversation.messages[-1]
            if (
                last_message.sender == self.role
                and last_message.content == response_content.response
            ):
                logger.warning(
                    f"Guardrail triggered: Agent attempted to repeat the last message: '{response_content.response}'"
                )
                return None

        # Use current timestamp for all messages
        response_timestamp = datetime.now()

        return Message(
            sender=self.role, content=response_content.response, timestamp=response_timestamp
        )

    def _format_examples(self, examples: list[Document]) -> list[BaseMessage]:
        """
        Formats the retrieved few-shot example Documents into a list of LangChain messages.
        Each Document's page_content (history, typically customer's turn) becomes a HumanMessage,
        and the metadata (next agent message) becomes an embedded agent response in the conversation history.
        """
        formatted_messages: list[BaseMessage] = []
        for index, doc in enumerate(examples):
            try:
                # Example agent response from metadata
                metadata = doc.metadata
                # Validate essential keys before creating Message object
                if not all(
                    key in metadata
                    for key in ["next_message_content", "full_conversation"]
                ):
                    logger.warning(f"Skipping few-shot example due to missing essential metadata: {metadata} in RagAgent.")
                    continue

                full_conversation = str(metadata["full_conversation"])
                agent_response = str(metadata["next_message_content"])

                # Replace the agent's turn in the conversation with the desired format
                agent_turn_str = f"Agent: {agent_response}"
                if agent_turn_str in full_conversation:
                    modified_conversation = full_conversation.replace(
                        agent_turn_str, f"**Current agent response**: {agent_response}"
                    )
                else:
                    logger.warning(
                        f"Could not find agent turn '{agent_turn_str}' in full conversation. Using original."
                    )
                    modified_conversation = full_conversation

                # Add indexing for the examples
                formatted_messages.append(HumanMessage(content=f"Example {index + 1}:"))
                formatted_messages.append(HumanMessage(content=modified_conversation))
                
            except Exception as e:
                logger.error(f"Error processing few-shot example in RagAgent: {e}")
                continue
        return formatted_messages

    @staticmethod
    async def _get_few_shot_examples(conversation_history: Sequence[Message], vector_store: VectorStore, k: int = 3) -> list[Document]:
        return await indexing_and_retrieval.get_similar_examples_for_next_message_role(
            conversation_history=conversation_history,
            vector_store=vector_store,
            k=k,
            target_role=ParticipantRole.AGENT
        )


@frozen
class RagAgentFactory(AgentFactory):
    """Factory for creating RAG-based agent participants."""
    
    model: BaseChatModel
    agent_vector_store: VectorStore
    agent_config: AgentConfig | None = None
    
    def create_participant(self) -> RagAgent:
        """Create a RAG agent participant.
        
        Returns:
            RagAgent instance configured with the vector store
        """
        return RagAgent(
            agent_vector_store=self.agent_vector_store,
            model=self.model
        )
