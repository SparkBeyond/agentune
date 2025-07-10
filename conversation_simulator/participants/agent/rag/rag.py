"""RAG-based agent participant implementation."""

from __future__ import annotations

import logging
import random
from datetime import datetime

from attrs import field, frozen
import attrs
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from ....models import Conversation, Message
from ....rag import indexing_and_retrieval
from ..base import Agent, AgentFactory
from ..config import AgentConfig
from .prompt import AGENT_PROMPT, AgentResponse


logger = logging.getLogger(__name__)


@frozen
class RagAgent(Agent):
    """RAG LLM-based agent participant.
    
    Uses a language model with Retrieval Augmented Generation
    to generate agent responses.
    """

    agent_vector_store: VectorStore
    model: BaseChatModel
    seed: int = 0
    intent_description: str | None = None # Store intent
    llm_chain: Runnable = field(init=False)
    _random: random.Random = field(init=False, repr=False)

    @llm_chain.default
    def _create_llm_chain(self) -> Runnable:
        """Creates the LangChain Expression Language (LCEL) chain for the agent."""
        # Use the imported AGENT_PROMPT from prompt.py
        # If there's an intent description, we can modify the system message
        prompt = AGENT_PROMPT
        
        # Return the runnable chain with the imported prompt
        return prompt | self.model | PydanticOutputParser(pydantic_object=AgentResponse)
    
    @_random.default
    def _create_random(self) -> random.Random:
        """Create a random number generator with the specified seed."""
        return random.Random(self.seed)

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

        # 2. Calculate probability of returning a message for the target role
        probability = await indexing_and_retrieval.probability_of_next_message_for(
            role=self.role,
            similar_docs=few_shot_examples
        )

        # 3. Examples selection and formatting
        # Select up to 5 most relevant examples
        few_shot_examples = few_shot_examples[:5]

        # Format few-shot examples and history for the prompt template
        formatted_examples = self._format_examples(few_shot_examples)
        formatted_current_conversation = indexing_and_retrieval._format_conversation_history(conversation.messages)

        # 4. Intent statement
                # Add the goal line to the conversation if there's an intent
        if self.intent_description:
            goal_line = f"This conversation was initiated by agent with the following intent:\n{self.intent_description}"
        else:
            goal_line = ""

        # 5. Chain execution
        chain_input = {
            "examples": formatted_examples,
            "current_conversation": formatted_current_conversation,
            "probability": probability,
            "goal_line": goal_line
        }
        response: AgentResponse = await self.llm_chain.ainvoke(chain_input)

        log_message = f"{self.role.value} - retrieved {len(few_shot_examples)} examples, probability : {probability}, decided to respond: {response.should_respond}"
        if response.should_respond:
            log_message = log_message + f"\n==> {response.response}"
        logger.debug(log_message)

        # 6. Process response
        # Check if the agent should respond
        if not response.should_respond:
            return None
        
        if response.response is None:
            logger.error("Agent response is None, even though should_respond is True!")
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
        Formats the retrieved few-shot example Documents into focused conversation examples.
        Uses focused conversation extraction to show only relevant context around the current message.
        """
        def _format_example(doc: Document, num: int) -> str:
            return f"Example conversation {num}:\n{doc.metadata['focused_conversation_part']}"

        conversations = [
            _format_example(doc, i+1) for i, (doc, _) in enumerate(examples)
        ]

        return "\n\n".join(conversations)


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
