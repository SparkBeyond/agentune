"""Prompts for the RAG customer participant."""

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# System prompt for the customer
CUSTOMER_SYSTEM_PROMPT = """You goal is to simulate a Customer in a text-based customer service conversation.
You'll be given a conversation history and a few examples of similar conversation states and their responses.
Your task is to generate a response that is similar in both style and content, based on the examples.

You will also be provided with a probability value which indicates the likelihood that the customer would respond at this point based on historical data.
Use this probability to inform your decision about whether the customer should respond or not.

Maintain consistent behavior across the current conversation. 
Pay attention to changes in tone and emotion in the example responses to understand how the current conversation should evolve."""

# Human message template for the customer
CUSTOMER_HUMAN_TEMPLATE = """Below are examples of similar conversation states and their responses:

{examples}

---

{goal_line}

Current conversation:
{current_conversation}

The probability that the customer would respond at this point (based on similar conversation patterns in the historical data) is: {probability}

Based on the examples, the current conversation, and the probability, answer the following questions:
1. Would the currently simulated customer respond next?
2. If so, what would the response be?
3. If not, why not?

Response format directions:
{format_instructions}
"""

# Create the prompt with both system and human messages
CUSTOMER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=CUSTOMER_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(CUSTOMER_HUMAN_TEMPLATE)
])