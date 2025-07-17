"""RAG-based agent participant implementation."""

from __future__ import annotations

import logging
from random import Random
from datetime import datetime
from attrs import field, frozen
import attrs
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from ....models import Conversation, Message
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
    seed: int = 0
    intent_description: str | None = None  # Store intent
    llm_chain: Runnable = field(init=False)
    _random: Random = field(init=False, repr=False)

    @llm_chain.default
    def _create_llm_chain(self) -> Runnable:
        """Creates the LangChain Expression Language (LCEL) chain for the agent."""
        # Use the imported AGENT_PROMPT from prompt.py
        # If there's an intent description, we can modify the system message
        prompt = AGENT_PROMPT

        # Return the runnable chain with the imported prompt
        return prompt | self.model | PydanticOutputParser(pydantic_object=AgentResponse)

    @_random.default
    def _create_random(self) -> Random:
        """Create a random number generator with the specified seed."""
        return Random(self.seed)

    def with_intent(self, intent_description: str) -> RagAgent:
        """Return a new RagAgent instance with the specified intent."""
        return attrs.evolve(self, intent_description=intent_description)

    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate next agent message using RAG LLM approach."""

        # 1. Retrieval
        few_shot_examples: list[tuple[Document, float]] = await indexing_and_retrieval.get_few_shot_examples(
            conversation_history=conversation.messages,
            vector_store=self.agent_vector_store,
            k=20
        )
        if not conversation.agent_messages:
            # If this is the first message, select the closest example
            few_shot_examples = [few_shot_examples[0]] if few_shot_examples else []
        else:
            # Select up to 5 randomly chosen examples
            few_shot_examples = self._random.sample(few_shot_examples, min(5, len(few_shot_examples)))

        # 2. Augmentation
        # Format few-shot examples and history for the prompt template
        formatted_examples = indexing_and_retrieval.format_examples(few_shot_examples)

        # Intent statement
        # Add the goal line to the conversation if there's an intent
        if self.intent_description:
            goal_line = f"This conversation was initiated by agent with the following intent:\n{self.intent_description}"
        else:
            goal_line = ""

        # 3. Generation
        formatted_current_convo = indexing_and_retrieval.format_conversation(conversation.messages)
        response_content: AgentResponse = await self.llm_chain.ainvoke({
            "examples": formatted_examples,
            "current_conversation": formatted_current_convo,
            "goal_line": goal_line
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
                logger.debug(
                    f"Guardrail triggered: Agent attempted to repeat the last message: '{response_content.response}'"
                )
                return None

        # Use current timestamp for all messages
        response_timestamp = datetime.now()

        return Message(
            sender=self.role, content=response_content.response, timestamp=response_timestamp
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
            model=self.model,
        )
