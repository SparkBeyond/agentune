"""Prompts for the RAG customer participant."""

from ._customer_response import CustomerResponse

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


# System prompt for the customer first message
CUSTOMER_FIRST_MESSAGE_SYSTEM_PROMPT = f"""Your goal is to simulate a Customer's first message in a text-based customer service conversation.

You'll be given ONE example of a similar conversation. Your task is to mimic that customer as closely as possible to capture their uniqueness and directness.

Simple decision flow:
1. Look at the example customer - how do they behave? Do they respond or not?
2. If they don't respond - don't respond
3. If they do respond:
   - If their content fits the current situation: mimic both style AND content
   - If their content doesn't fit: mimic their STYLE while adapting to the current conversation

Avoid generalizing. Adopt specific behavior and content as closely as possible.

Output format:
The output should be formatted as a JSON instance that conforms to the JSON schema below. **The `reasoning` field must be the first key in the object.**

{PydanticOutputParser(pydantic_object=CustomerResponse).get_format_instructions()}
"""

# Human message template for the customer first message
CUSTOMER_FIRST_MESSAGE_HUMAN_TEMPLATE = """Follow the simple decision flow above.

---

Example customer behavior to mimic:
{examples}

---

Additional context:
{goal_line}

---

Current conversation:
{current_conversation}

---

Output only the JSON object with your reasoning and decision.
"""

# Create the prompt with both system and human messages
CUSTOMER_FIRST_MESSAGE_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=CUSTOMER_FIRST_MESSAGE_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(CUSTOMER_FIRST_MESSAGE_HUMAN_TEMPLATE)
])