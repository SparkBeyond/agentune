"""Prompts for adversarial testing conversation simulation quality."""


from ...models.conversation import Conversation


def format_conversation(conversation: Conversation) -> str:
    """Format a conversation for display in a prompt.
    
    Args:
        conversation: The conversation to format
        
    Returns:
        A formatted string representation of the conversation
    """
    formatted_messages = []
    for message in conversation.messages:
        role = "Customer" if message.sender.name == "CUSTOMER" else "Agent"
        formatted_messages.append(f"{role}: {message.content}")
    
    return "\n".join(formatted_messages)


SYSTEM_PROMPT = """You are tasked with distinguishing between a real conversation and an auto-generated conversation.
Based solely on the content, context flow, and natural language patterns, and similarity with example real conversations, determine which conversation is real and which is auto-generated.

Reasoning flow:
Focus on Customer messages - in real conversations it's a human customer, in generated an LLM was used. 
Analyze the example real conversations for:
1. Natural flow of dialogue
2. Language style
3. Messages length
4 Appropriate responses to questions/statements
5. Presence of natural human errors or patterns
6. Contextual understanding

Then apply the same analysis to the compared conversations
Which of the compared conversations follows the same patterns?

Return your answer in the following JSON format, provide detailed reasoning, make sure "reasoning" field appears first
{
  "reasoning": "your reasoning"
  "real_conversation": "A" or "B"
}"""

HUMAN_PROMPT_TEMPLATE = """Here are some examples of real conversations:

{examples}

Conversation A:\n{conversation_a}\n\nConversation B:\n{conversation_b}"""


def format_examples(conversations: tuple[Conversation, ...]) -> str:
    formatted_examples = ""
    for i, conversation in enumerate(conversations, 1):
        formatted_examples += f"Example {i}:\n{format_conversation(conversation)}\n\n"
    return formatted_examples.rstrip()


def create_comparison_prompt_inputs(
    real_conversation: Conversation,
    simulated_conversation: Conversation,
    formatted_examples: str,
    is_real_a: bool,
) -> dict[str, str]:
    """Create inputs for the conversation comparison prompt.

    Args:
        real_conversation: The real conversation.
        simulated_conversation: The simulated conversation.
        formatted_examples: Formatted examples of real conversations.
        is_real_a: Whether the real conversation should be labeled 'A' (True) or 'B' (False).

    Returns:
        A dictionary with formatted conversations for the prompt.
    """
    if is_real_a:
        conv_a = format_conversation(real_conversation)
        conv_b = format_conversation(simulated_conversation)
    else:
        conv_a = format_conversation(simulated_conversation)
        conv_b = format_conversation(real_conversation)

    return {"conversation_a": conv_a, "conversation_b": conv_b, "examples": formatted_examples}
