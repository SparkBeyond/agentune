"""Zero-shot agent participant implementation."""

from datetime import datetime

import attrs
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from ....models.conversation import Conversation
from ....models.message import Message
from ..config import AgentConfig
from ..base import Agent, AgentFactory
from .prompts import AgentPromptBuilder


@attrs.frozen
class ZeroShotAgent(Agent):
    """Zero-shot LLM-based agent participant.
    
    Uses a language model to generate agent responses without
    fine-tuning or few-shot examples.
    
    This class is immutable - use with_intent() to create new instances
    with different intents.
    """
    
    agent_config: AgentConfig
    model: BaseChatModel
    intent_description: str | None = None
    
    # Private fields for internal state
    _prompt_builder: AgentPromptBuilder = attrs.field(factory=AgentPromptBuilder, init=False)
    _output_parser: StrOutputParser = attrs.field(factory=StrOutputParser, init=False)
    
    def with_intent(self, intent_description: str) -> 'ZeroShotAgent':
        """Return a new agent instance with the specified intent.
        
        Args:
            intent_description: Natural language description of the agent's goal/intent
            
        Returns:
            New ZeroShotAgent instance with the intent installed
        """
        return attrs.evolve(self, intent_description=intent_description)
    
    def _get_chain(self) -> Runnable:
        """Create the LangChain processing chain."""
        prompt_template = self._prompt_builder.build_chat_template(
            agent_config=self.agent_config,
            intent_description=self.intent_description
        )
        return prompt_template | self.model | self._output_parser
    
    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate next agent message using zero-shot LLM approach.
        
        Args:
            conversation: Current conversation history
            
        Returns:
            Generated message or None if conversation should end
        """
        # Convert conversation to messages for the chain
        conversation_history = conversation.to_langchain_messages()
        
        # Use the chain to get response
        chain = self._get_chain()
        agent_response = await chain.ainvoke({
            "conversation_history": conversation_history
        })
        
        # If message is empty, do not respond
        if not agent_response.strip():
            return None
        else:
            # Use current timestamp for all messages
            response_timestamp = datetime.now()
            
            return Message(
                sender=self.role,
                content=agent_response.strip(),
                timestamp=response_timestamp
            )

@attrs.frozen
class ZeroShotAgentFactory(AgentFactory):
    """Factory for creating zero-shot agent participants.
    
    Args:
        model: LangChain chat model for agent responses
        agent_config: Configuration for the agent's role and company context
    """

    model: BaseChatModel
    agent_config: AgentConfig
        
    def create_participant(self) -> ZeroShotAgent:
        """Create a zero-shot agent participant.
        
        Returns:
            ZeroShotAgent instance
        """
        return ZeroShotAgent(
            model=self.model,
            agent_config=self.agent_config
        )
