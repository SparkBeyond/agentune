"""Agent prompt generation for LLM interactions."""

from __future__ import annotations

import attrs
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from ....models.conversation import Conversation
from ....models.roles import ParticipantRole
from ..config import AgentConfig


@attrs.define
class AgentPromptBuilder:
    """Builds prompts for agent LLM interactions."""

    def build_chat_template(
        self,
        agent_config: AgentConfig,
        intent_description: str | None = None
    ) -> ChatPromptTemplate:
        """Build chat prompt template for LLM chain.
        
        Args:
            agent_config: Configuration for the agent
            intent_description: Optional natural language description of agent's goal/intent
            
        Returns:
            ChatPromptTemplate for use in LangChain chains
        """
        system_prompt = self._build_system_prompt(agent_config, intent_description)
        
        # Create template with system message and conversation history placeholder
        template_messages = [
            ("system", system_prompt),
            ("placeholder", "{conversation_history}"),
        ]
        
        return ChatPromptTemplate.from_messages(template_messages)

    def conversation_to_messages(self, conversation: Conversation) -> list[BaseMessage]:
        """Convert conversation history to LangChain messages.
        
        Args:
            conversation: Current conversation history
            
        Returns:
            List of LangChain BaseMessage objects
        """
        messages: list[BaseMessage] = []
        
        # Add conversation history
        for msg in conversation.messages:
            if msg.sender == ParticipantRole.CUSTOMER:
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))
        
        return messages

    def build_chat_messages(
        self,
        agent_config: AgentConfig,
        conversation: Conversation, 
        intent_description: str | None = None,
        format_instructions: str | None = None
    ) -> list[BaseMessage]:
        """Build chat messages for LLM completion.
        
        Args:
            agent_config: Configuration for the agent
            conversation: Current conversation history
            intent_description: Optional natural language description of agent's goal/intent
            format_instructions: Optional format instructions for output
            
        Returns:
            List of LangChain BaseMessage objects
        """
        messages: list[BaseMessage] = [SystemMessage(content=self._build_system_prompt(agent_config, intent_description))]
        
        # Add conversation history
        for msg in conversation.messages:
            if msg.sender == ParticipantRole.CUSTOMER:
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))
        
        return messages
    
    def _build_system_prompt(self, agent_config: AgentConfig, intent_description: str | None = None) -> str:
        """Build the system prompt for the agent.
        
        Args:
            agent_config: Configuration for the agent
            intent_description: Optional natural language description of agent's goal/intent
            
        Returns:
            System prompt string
        """
        prompt_parts = [
            f"You are a {agent_config.agent_role} at {agent_config.company_name}.",
            "",
            f"Company: {agent_config.company_name}",
            f"About the company: {agent_config.company_description}",
            "",
            "Guidelines:",
            "- Be helpful, professional, and courteous",
            "- Focus on resolving customer issues efficiently",
            "- Use natural, conversational language",
            "- Keep responses concise and relevant",
            "- Stay in character as the company representative",
        ]
        
        # Add intent if provided
        if intent_description:
            prompt_parts.extend([
                "",
                f"Your goal for this conversation: {intent_description}",
            ])
        
        return "\n".join(prompt_parts)
