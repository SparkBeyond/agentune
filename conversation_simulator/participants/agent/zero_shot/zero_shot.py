"""Zero-shot agent participant implementation."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from ....models.conversation import Conversation
from ....models.intent import Intent
from ....models.message import Message
from ....models.roles import ParticipantRole
from ..config import AgentConfig
from ..base import Agent
from .prompts import AgentPromptBuilder


class ZeroShotAgent(Agent):
    """Zero-shot LLM-based agent participant.
    
    Uses a language model to generate agent responses without
    fine-tuning or few-shot examples.
    """
    
    def __init__(self, agent_config: AgentConfig, model: BaseChatModel, intent: Intent | None = None) -> None:
        """Initialize zero-shot agent.
        
        Args:
            agent_config: Configuration for the agent
            model: LangChain chat model instance (e.g., ChatOpenAI, ChatAnthropic, etc.)
            intent: Optional agent intent/goal
        """
        self.agent_config = agent_config
        self.model = model
        self.intent = intent
        self.prompt_builder = AgentPromptBuilder()
        self.output_parser = StrOutputParser()
        
        # Build the chain: prompt_template | model | parser
        self.prompt_template = self.prompt_builder.build_chat_template(
            agent_config=self.agent_config,
            intent=self.intent
        )
        self.chain = self.prompt_template | self.model | self.output_parser
    
    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate next agent message using zero-shot LLM approach.
        
        Args:
            conversation: Current conversation history
            
        Returns:
            Generated message or None if conversation should end
        """
        # Convert conversation to messages for the chain
        conversation_history = self.prompt_builder.conversation_to_messages(conversation)
        
        # Use the chain to get response
        agent_response = await self.chain.ainvoke({
            "conversation_history": conversation_history
        })
        
        # If message is empty, end conversation
        if not agent_response.strip():
            return None
        else:
            # Generate fake timestamp: 3-20 seconds after last message
            if conversation.messages:
                last_timestamp = conversation.messages[-1].timestamp
                delay_seconds = random.randint(3, 20)
                response_timestamp = last_timestamp + timedelta(seconds=delay_seconds)
            else:
                # If no messages yet, use current time
                response_timestamp = datetime.now()
                
            return Message(
                sender=ParticipantRole.AGENT,
                content=agent_response.strip(),
                timestamp=response_timestamp
            )
