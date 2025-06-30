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
Based solely on the content, context flow, and natural language patterns, determine which conversation is real and which is auto-generated.

Analyze the two conversations for:
1. Natural flow of dialogue
2. Consistency in language style
3. Appropriate responses to questions/statements
4. Presence of natural human errors or patterns
5. Contextual understanding

Return your answer in the following JSON format without any additional explanations:
{{
  "real_conversation": "A" or "B"
}}"""

HUMAN_PROMPT_TEMPLATE = """Conversation A:\n{conversation_a}\n\nConversation B:\n{conversation_b}"""


def create_system_prompt(
    example_conversations: tuple[Conversation, ...] | None = None,
) -> str:
    """Creates a system prompt that includes example conversations.

    Args:
        example_conversations: Optional list of example conversations to include.

    Returns:
        The system prompt with examples incorporated.
    """
    if not example_conversations:
        return SYSTEM_PROMPT

    # Create a prompt with examples
    examples_formatted = "\n\nHere are some examples of real conversations:\n\n"
    for i, example in enumerate(example_conversations, 1):
        formatted_conversation = format_conversation(example)
        examples_formatted += f"Example {i}:\n{formatted_conversation}\n\n"

    # Return the system prompt with examples
    return SYSTEM_PROMPT + examples_formatted.rstrip()


def create_comparison_prompt_inputs(
    real_conversation: Conversation,
    simulated_conversation: Conversation,
    is_real_a: bool,
) -> dict[str, str]:
    """Create inputs for the conversation comparison prompt.

    Args:
        real_conversation: The real conversation.
        simulated_conversation: The simulated conversation.
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

    return {"conversation_a": conv_a, "conversation_b": conv_b}
