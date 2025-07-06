"""Prompts for the RAG customer participant."""

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


class CustomerResponse(BaseModel):
    """Customer's response (if relevant) with reasoning."""
    
    reasoning: str = Field(
        description="Detailed reasoning for why the customer would respond or not, and what the response would be"
    )
    should_respond: bool = Field(
        description="Whether the customer should respond at this point"
    )
    response: str | None = Field(
        default=None,
        description="Response content, or null if should_respond is false"
    )


# System prompt for the customer
CUSTOMER_SYSTEM_PROMPT = f"""Your goal is to simulate a Customer in a text-based customer service conversation.
You'll be given a conversation history and a few examples of similar conversation states and their responses.
Your task is to generate a response that is similar in both style and content, based on the examples.

You will also be provided with a probability value which indicates the likelihood that the customer would respond at this point based on historical data.
Use this probability to inform your decision about whether the customer should respond or not.

Maintain consistent behavior across the current conversation. 
Pay attention to changes in tone and emotion in the example responses to understand how the current conversation should evolve.
Avoid giving too generic answers. Pick the example for which customer's tone fits the current customer the most, and use it as a source for the style and content for the next message. If that customer did not respond - mimic it.

---

Reasoning flow:
1. Pick an example with a customer most similar in style to the current customer. Explain your choice.
2. Use that customer as a main source of inspiration for your decision.
3. Act as that customer. 
4. Given the current state of the conversation would that customer answer?
5. If not - prefer not to answer. If yes - what style would they choose?
6. What would they answer? Remember to keep it consistent with the current conversation

---

### Output format

The output should be formatted as a JSON instance that conforms to the JSON schema below. **The `reasoning` field must be the first key in the object.**

{PydanticOutputParser(pydantic_object=CustomerResponse).get_format_instructions()}
"""

# Human message template for the customer
CUSTOMER_HUMAN_TEMPLATE = """Instructions: follow the System Guidance above when deciding if the customer should answer at this point, and what they would answer.

--- 

Below are examples of similar conversation states and their responses:

{examples}

---

Current conversation:
{current_conversation}

---

Additional information about the current conversation:

{goal_line}

The probability that the customer would respond at this point (based on similar conversation patterns in the historical data) is estimated at: {probability}

---

Output only the JSON object, following the workflow described in the System Guidance.
"""

# Create the prompt with both system and human messages
CUSTOMER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=CUSTOMER_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(CUSTOMER_HUMAN_TEMPLATE)
])