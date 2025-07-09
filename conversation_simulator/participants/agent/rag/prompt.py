"""Prompts for the RAG agent participant."""

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


class AgentResponse(BaseModel):
    """Agent's response (if relevant) with reasoning."""
    
    reasoning: str = Field(
        description="Detailed reasoning for why the agent would respond or not, and what the response would be"
    )
    should_respond: bool = Field(
        description="Whether the agent should respond at this point"
    )
    response: str | None = Field(
        default=None,
        description="Response content, or null if should_respond is false"
    )


# System prompt for the agent
AGENT_SYSTEM_PROMPT = f"""You goal is to simulate an Agent in a text-based conversation.
You'll be given a conversation history and a few examples of similar conversation states and their responses.
Your task is to generate a response that is similar in both style and content, based on the examples.

Important! Following the same style is **mandatory**!

---

Decision-making guidance

You will also be provided with a probability value which indicates the likelihood that the agent would respond at this point based on historical data.
Use this probability to inform your decision about whether the agent should respond or not.
Additionally, consider the content of the current conversation and the examples provided, to determine if the agent should respond.

---

Reasoning directions

Based on the examples, the current conversation, and the probability, answer the following questions:
1. Would a real agent respond next?
2. If so, what would the response be?
3. If not, why not?

---

### Output format

The output should be formatted as a JSON instance that conforms to the JSON schema below. **The `reasoning` field must be the first key in the object.**

{PydanticOutputParser(pydantic_object=AgentResponse).get_format_instructions()}

"""

# Human message template for the agent
AGENT_HUMAN_TEMPLATE = """Instructions: follow the System Guidance above when deciding if the agent should answer at this point, and what they would answer.

Below are examples of similar conversation states and their responses:

{examples}

---

Current conversation:
{current_conversation}

---

Additional information about the current conversation:

{goal_line}

The probability that the agent would respond at this point (based on similar conversation patterns in the historical data) is estimated at: {probability}

---

Output only the JSON object, following the workflow described in the System Guidance.
"""

# Create the prompt with both system and human messages
AGENT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=AGENT_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(AGENT_HUMAN_TEMPLATE)
])
