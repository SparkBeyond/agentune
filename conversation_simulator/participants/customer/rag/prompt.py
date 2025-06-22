"""Prompts for the RAG customer participant."""

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# System prompt for the customer
CUSTOMER_SYSTEM_PROMPT = """You are simulating a customer in a text-based customer service conversation."""

# Human message template for the customer
CUSTOMER_HUMAN_TEMPLATE = """Below are examples of similar conversation states and their responses:

{examples}

# Current conversation:
{current_conversation}

Generate a response as a customer that is natural and informative. Your response should reflect how real people communicate in customer service interactions.

CRITICAL REQUIREMENTS:

1. APPROPRIATE LENGTH: Use 1-3 sentences that provide sufficient information
2. CONVERSATIONAL STYLE: Use contractions (I'm, don't, can't) and natural language
3. MEANINGFUL CONTENT: Include enough detail to advance the conversation

As a customer:
- Clearly express your question or concern with relevant details
- Use natural, everyday language ("I need help with my order" not "I would like assistance regarding my purchase")
- Include context that helps the agent understand your situation
- Your goal in this conversation is: {customer_goal}

Your response should be realistic and contain enough substance to maintain a meaningful conversation."""

# Create the prompt with both system and human messages
CUSTOMER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=CUSTOMER_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(CUSTOMER_HUMAN_TEMPLATE)
])