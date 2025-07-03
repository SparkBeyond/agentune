"""Prompts for RAG-based outcome detection."""


from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ...models.conversation import Conversation
from ...models.intent import Intent
from ...models.outcome import Outcomes

# System prompt for outcome detection
SYSTEM_PROMPT_TEMPLATE = """You are an expert at analyzing conversations and detecting when specific outcomes have been reached.

If an outcome has been reached, set 'detected' to true and specify the exact outcome name in 'outcome'.
If no outcome has been reached yet, set 'detected' to false and 'outcome' to null.
Always provide detailed reasoning for your decision.
"""

# Human message template for outcome detection
HUMAN_PROMPT_TEMPLATE = """I need you to analyze if a conversation has reached a specific outcome.

POSSIBLE OUTCOMES:
{outcomes_str}

Here are some example completed conversations for reference:

{examples_text}

Here is the conversation to analyze:

{conversation_text}

This conversations was initiated by {intent_role} 
Intent: {intent_description}

Has this conversation reached one of the defined outcomes? If so, which one? Provide detailed reasoning for your analysis.

Format your response as a JSON object with the following structure:
{format_instructions}
"""


def build_system_prompt(output_parser: PydanticOutputParser) -> str:
    """Build the minimal system prompt for outcome detection.
        
    Args:
        output_parser: The output parser for formatting instructions
        
    Returns:
        System prompt string
    """
    format_instructions = output_parser.get_format_instructions()
    return SYSTEM_PROMPT_TEMPLATE.format(format_instructions=format_instructions)


def build_human_prompt(
    format_instructions: str,
    conversation: Conversation, 
    examples: list[tuple[Document, float]],
    intent: Intent,
    possible_outcomes: Outcomes
) -> str:
    """Build the human prompt containing the conversation and examples.
    
    Args:
        format_instructions: Instructions for formatting the response
        conversation: The conversation to analyze
        examples: List of similar conversations as (Document, score) tuples
        intent: The conversation intent/goal
        possible_outcomes: Possible outcomes to detect
        
    Returns:
        Human prompt string
    """
    # Format outcomes
    outcomes_str = "\n".join([
        f"- {outcome.name}: {outcome.description}" 
        for outcome in possible_outcomes.outcomes
    ])
    
    # Format the current conversation
    conversation_text = "\n".join([
        f"{message.sender}: {message.content}"
        for message in conversation.messages
    ])
    
    # Format the examples
    examples_text = format_examples(examples)
    
    return HUMAN_PROMPT_TEMPLATE.format(
        format_instructions=format_instructions,
        intent_role=intent.role.title(),
        intent_description=intent.description,
        outcomes_str=outcomes_str,
        examples_text=examples_text,
        conversation_text=conversation_text
    )


def format_examples(examples: list[tuple[Document, float]]) -> str:
    """Format retrieved examples for the prompt.
    
    Args:
        examples: Retrieved similar conversation examples
        
    Returns:
        Formatted examples string with outcome annotations when available
    """
    formatted_examples = []
    
    for i, (doc, _) in enumerate(examples):
        conversation_text = doc.metadata.get('full_conversation', '')
        outcome_info = doc.metadata.get('outcome', 'No outcome information available')
        
        example = f"Example {i+1}:\n{conversation_text}\n\nOutcome: {outcome_info}"
        formatted_examples.append(example)
    
    return "\n\n".join(formatted_examples)


# Create a prompt template that can be reused
def create_prompt_template(output_parser: PydanticOutputParser) -> ChatPromptTemplate:
    """Create a ChatPromptTemplate with system and human messages.
    
    Args:
        output_parser: The output parser for formatting instructions
        
    Returns:
        A ready-to-use ChatPromptTemplate
    """
    system_content = build_system_prompt(output_parser)
    
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        HumanMessagePromptTemplate.from_template(HUMAN_PROMPT_TEMPLATE)
    ])
