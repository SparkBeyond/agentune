"""Prompts for RAG-based outcome detection."""

from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ...models.message import Message
from ...rag import indexing_and_retrieval
from ...util.structure import converter


class OutcomeDetectionResult(BaseModel):
    """Result of outcome detection with reasoning."""
    
    reasoning: str = Field(
        description="Explanation of why this outcome was detected. Alternatively, explanation of why no outcome was detected"
    )
    detected: bool = Field(
        description="Whether a specific outcome was detected in the conversation"
    )
    outcome: str | None = Field(
        default=None,
        description="Name of the detected outcome, or null if no outcome was detected"
    )


# System prompt for outcome detection
SYSTEM_PROMPT = f"""You are an expert at analyzing conversations and detecting when specific outcomes have been reached.

**Key guidance about evidence sources**

1. The conversation text itself may *not* always provide enough information to assign an outcome confidently (e.g., relevant events—such as a purchase, support-ticket resolution, or account change—can occur after the conversation ends or without any additional messages).
2. Therefore, you **must leverage the provided example conversations as your primary external evidence**.  Compare the structure, tone, key phrases, and context of the current conversation with those examples.  When the current conversation is highly similar to one or more examples, assume the same outcome *unless* there is clear contradictory evidence in the conversation.
3. Do not blindly copy a majority label; reason about *why* a specific example (or group of examples) justifies the chosen outcome.
4. If neither the conversation **nor** the examples give sufficient evidence, respond with `"detected": false` and `"outcome": null`.

---

If an outcome has been reached, set `detected` to **true** and specify the exact outcome name in `outcome`.
If no outcome has been reached yet, set `detected` to **false** and `outcome` to **null**.
Always provide detailed reasoning for your decision.

---

### Output format

Respond with a single JSON object that conforms to the following schema.  **The `reasoning` field must be the first key in the object.**

{PydanticOutputParser(pydantic_object=OutcomeDetectionResult).get_format_instructions()}

---

### Recommended reasoning workflow (chain‑of‑thought)

1. **Extract clues** from the conversation that might indicate an outcome (purchase intent, discount acceptance, explicit refusal, etc.).
2. **Retrieve parallels**: list the example conversations that most closely match those clues.
3. **Compare & decide**: explain why the chosen examples support a particular outcome (or why evidence is insufficient).
4. **Produce JSON** as specified.
"""


# Human message template for outcome detection
HUMAN_PROMPT_TEMPLATE = """Instructions: follow the System Guidance above when deciding the conversations outcome.

---

POSSIBLE OUTCOMES:
{outcomes_str}

---

Here are some example completed conversations for reference.
Those conversations have been picked due to their similarity to currently analyzed conversations, and should be used as a source to understanding what should be it's outcome.

{examples_text}

---

Additional information about the analyzed conversation:

This conversation was initiated by {intent_role} 
Intent: {intent_description}

---

Here is the conversation to analyze:

{conversation_text}

---

Output only the JSON object, following the workflow described in the System Guidance.
"""


def format_examples(examples: list[tuple[Document, float]]) -> str:
    """Format retrieved examples for the prompt.
    
    Args:
        examples: Retrieved similar conversation examples
        
    Returns:
        Formatted examples string with outcome annotations when available
    """
    formatted_examples = []
    
    for i, (doc, _) in enumerate(examples):
        messages_data = doc.metadata.get('full_conversation', '')
        messages = converter.structure(messages_data, list[Message])
        conversation_text = indexing_and_retrieval.format_conversation(messages)

        outcome_info = doc.metadata.get('outcome', 'No outcome information available')
        
        example = f"Example {i+1}:\n{conversation_text}\n\nOutcome: {outcome_info}"
        formatted_examples.append(example)
    
    return "\n\n".join(formatted_examples)


# Create a prompt template that can be reused
OUTCOME_DETECTION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(HUMAN_PROMPT_TEMPLATE)
])
