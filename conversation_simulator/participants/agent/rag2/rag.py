"""RAG-based agent participant implementation."""

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
from ..base import Agent, AgentFactory
from .prompt import AGENT_PROMPT

logger = logging.getLogger(__name__)


class AgentResponse(BaseModel):
    """Agent's response (if relevant) with reasoning."""
    
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


class Rag2Agent(Agent):
    """RAG LLM-based agent participant.
    
    Uses a language model with Retrieval Augmented Generation
    to generate agent responses.
    """

    def __init__(
        self,
        agent_vector_store: VectorStore,
        model: BaseChatModel,
        seed: int = 0
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
        self._output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
        self.llm_chain = self._create_llm_chain(model=model)
        self._random = random.Random(seed)
    
    def _create_llm_chain(self, model: BaseChatModel) -> Runnable:
        """Creates the LangChain Expression Language (LCEL) chain for the agent."""
        # Use the imported AGENT_PROMPT from prompt.py
        # If there's an intent description, we can modify the system message
        prompt = AGENT_PROMPT
        
        # Return the runnable chain with the imported prompt
        return prompt | model | self._output_parser

    def with_intent(self, intent_description: str) -> Rag2Agent:
        """Return a new Rag2Agent instance with the specified intent."""
        new_agent = Rag2Agent(
            agent_vector_store=self.agent_vector_store,
            model=self.model
        )
        new_agent.intent_description = intent_description
        return new_agent

    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate next agent message using RAG LLM approach."""
        # if not conversation.messages or conversation.messages[-1].sender != ParticipantRole.CUSTOMER:
        #     return None

        # 1. Retrieval
        few_shot_examples: list[tuple[Document, float]] = await index_by_prefix.get_few_shot_examples(
            conversation_history=conversation.messages,
            vector_store=self.agent_vector_store,
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
            goal_line = f"This conversation was initiated by agent with the following intent:\n{self.intent_description}"
        else:
            goal_line = ""

        # 3. Generation
        chain = self._create_llm_chain(model=self.model)
        response: AgentResponse = await chain.ainvoke({
            "examples": formatted_examples,
            "current_conversation": formatted_current_conversation,
            "goal_line": goal_line,
            "format_instructions": self._output_parser.get_format_instructions()
        })

        if not response.should_respond:
            logger.debug(f"Role: {self.role}, LLM decided not to respond: {response.reasoning}")
            return None

        if response.response is None:
            logger.error("Agent response is None, even though should_respond is True!")
            return None

        logger.debug(f"Role: {self.role}, LLM decided to respond: {response.reasoning}. Response: {response.response}")

        return Message(
            sender=self.role,
            content=response.response,  # Now guaranteed to be not None
            timestamp=current_timestamp
            )

    def _format_examples(self, examples: list[tuple[Document, float]]) -> str:
        """
        Formats the retrieved few-shot example Document into a LangChain message.
        Each Document's page_content (history, typically customer's turn) becomes a HumanMessage,
        and the metadata (next agent message) becomes an AIMessage.
        """
        conversations = [
            f"Example conversation {i+1}:\n{doc.metadata['full_conversation']}" 
            for i, (doc, _) in enumerate(examples)
        ]

        return "\n\n".join(conversations)


class Rag2AgentFactory(AgentFactory):
    """Factory for creating RAG-based agent participants."""
    
    def __init__(
        self, 
        model: BaseChatModel, 
        agent_vector_store: VectorStore
    ) -> None:
        """Initialize the factory.
        
        Args:
            model: LangChain chat model for agent responses
            agent_vector_store: Vector store containing agent message examples
        """
        self.model = model
        self.agent_vector_store = agent_vector_store
    
    def create_participant(self) -> Rag2Agent:
        """Create a RAG agent participant.
        
        Returns:
            Rag2Agent instance configured with the vector store
        """
        return Rag2Agent(
            agent_vector_store=self.agent_vector_store,
            model=self.model
        )
