"""Customer prompt generation for LLM interactions."""

import attrs
from langchain_core.prompts import ChatPromptTemplate


@attrs.define
class CustomerPromptBuilder:
    """Builds prompts for customer LLM interactions."""

    def build_chat_template(self, intent_description: str | None = None) -> ChatPromptTemplate:
        """Build chat prompt template for LLM chain.
        
        Args:
            intent_description: Optional natural language description of customer's goal/intent
            
        Returns:
            ChatPromptTemplate for use in LangChain chains
            
        Note:
            The template expects conversation history in LangChain message format.
            Use conversation.to_langchain_messages() to convert conversation history.
        """
        system_prompt = self._build_system_prompt(intent_description)
        
        # Create template with system message and conversation history placeholder
        template_messages = [
            ("system", system_prompt),
            ("placeholder", "{conversation_history}"),
        ]
        
        return ChatPromptTemplate.from_messages(template_messages)

    def _build_system_prompt(self, intent_description: str | None = None) -> str:
        """Build the system prompt for the customer.
        
        Args:
            intent_description: Optional natural language description of customer's goal/intent
            
        Returns:
            System prompt string
        """
        prompt_parts = [
            "You are a customer engaging with a company's support or sales team.",
            "",
            "Customer Profile:",
            "- You have a genuine need or question about the company's products/services",
            "- You communicate naturally and conversationally",
            "- You may ask clarifying questions or express concerns",
            "- You appreciate helpful, clear responses",
            "",
            "Guidelines:",
            "- Be authentic and realistic in your responses",
            "- Ask follow-up questions when you need clarification",
            "- Express satisfaction when your needs are met",
            "- Show reasonable patience but escalate if frustrated",
            "- Use natural, conversational language",
            "- Stay focused on your needs and concerns",
        ]
        
        # Add intent if provided
        if intent_description:
            prompt_parts.extend([
                "",
                f"Your specific goal or need for this conversation: {intent_description}",
            ])
        
        return "\n".join(prompt_parts)
