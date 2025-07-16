"""Prompt templates and utilities for intent extraction."""


from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from ..models.conversation import Conversation


def format_conversation(conversation: Conversation) -> str:
    """Format a conversation into a string with sender and content.
    
    Args:
        conversation: The conversation to format
        
    Returns:
        Formatted conversation string with sender and content
    """
    return "\n".join(
        f"{msg.sender.value.upper()}: {msg.content}"
        for msg in conversation.messages
    )


# System prompt for intent extraction
INTENT_EXTRACTION_SYSTEM_PROMPT = """You are an expert at analyzing conversations and identifying user intents.

Your task is to analyze the conversation and determine:
1. Who was the first to express a clear intent (CUSTOMER or AGENT)
2. What their specific intent/goal is

Important guidelines for identifying intent:
- The first speaker might just be making a greeting (e.g., "How can I help you?") rather than expressing intent
- Look for the first message that contains a clear purpose, request, or offer
- The intent could be expressed by either the customer or the agent
- If an AGENT initiates with a clear offer, proposal, or request for action, that should be considered the intent
- A CUSTOMER asking for information in response to an AGENT's offer is not a new intent
- Focus on who is driving the conversation purpose

Common intent categories include but are not limited to:
- IT Support (e.g., reporting technical issues, access problems)
- Sales/Purchasing (e.g., buying a product, inquiring about pricing, making an offer)
- Support/Help (e.g., account problems, service issues)
- Information Request (e.g., asking specific questions, gathering details)
- Feedback/Complaint (e.g., providing feedback, making a complaint)
- Scheduling/Booking (e.g., making appointments, reservations)
- Proactive Offer (e.g., agent making a special offer or upgrade)

Be specific in describing the intent. For example, instead of just "IT issue",
specify the nature of the issue in detail (e.g., "cannot access email account since last week, using Outlook on Windows 10").

Examples:
- If an AGENT says "I'm calling about your car's extended warranty", the intent is from the AGENT
- If a CUSTOMER says "I need help with my password", the intent is from the CUSTOMER
- If an AGENT makes an offer and the CUSTOMER asks questions about it, the intent is still from the AGENT
"""

# Human prompt template for intent extraction
INTENT_EXTRACTION_HUMAN_TEMPLATE = """Analyze the following conversation and determine the intent:

{conversation}

Format your response according to the following guidelines:
{format_instructions}

Your analysis:"""


def create_intent_extraction_prompt(format_instructions: str) -> ChatPromptTemplate:
    """Create a chat prompt template for intent extraction.
    
    Args:
        format_instructions: Instructions for formatting the output
        
    Returns:
        Configured ChatPromptTemplate for intent extraction
    """
    system_prompt = SystemMessagePromptTemplate.from_template(INTENT_EXTRACTION_SYSTEM_PROMPT)
    human_prompt = HumanMessagePromptTemplate.from_template(
        INTENT_EXTRACTION_HUMAN_TEMPLATE,
        partial_variables={"format_instructions": format_instructions}
    )
    return ChatPromptTemplate.from_messages([system_prompt, human_prompt])
