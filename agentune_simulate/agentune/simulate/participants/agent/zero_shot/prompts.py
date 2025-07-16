"""Agent prompt generation for LLM interactions."""

import attrs
from langchain_core.prompts import ChatPromptTemplate

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
            
        Note:
            The template expects conversation history in LangChain message format.
            Use conversation.to_langchain_messages() to convert conversation history.
        """
        system_prompt = self._build_system_prompt(agent_config, intent_description)
        
        # Create template with system message and conversation history placeholder
        template_messages = [
            ("system", system_prompt),
            ("placeholder", "{conversation_history}"),
        ]
        
        return ChatPromptTemplate.from_messages(template_messages)

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
