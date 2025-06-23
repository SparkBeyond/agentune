"""Prompts for the RAG customer participant."""

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# System prompt for the customer
CUSTOMER_SYSTEM_PROMPT = """You are simulating a customer in a text-based customer service conversation. 

Pay close attention to the tone and emotion in the example responses. Match the level of frustration, politeness, and style from the examples in your own response.

If the examples show frustrated or upset customers, respond in a similarly frustrated tone. If the examples show polite customers, respond politely. Always match the emotional tone of the examples."""

# Human message template for the customer
CUSTOMER_HUMAN_TEMPLATE = """Below are examples of similar conversation states and their responses:

{examples}

# Current conversation:
{current_conversation}

Generate a natural response as a customer in this conversation.
{goal_line}

Your response should be realistic and contain enough substance to maintain a meaningful conversation.

Customer response:"""

# Create the prompt with both system and human messages
CUSTOMER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=CUSTOMER_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(CUSTOMER_HUMAN_TEMPLATE)
])