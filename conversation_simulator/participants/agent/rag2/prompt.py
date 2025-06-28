"""Prompts for the RAG agent participant."""

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# System prompt for the agent
AGENT_SYSTEM_PROMPT = """You goal is to simulate an Agent in a text-based conversation.
You'll be given a conversation history and a few examples of similar conversation states and their responses.
Your task is to generate a response that is similar in both style and content, based on the examples."""

# Human message template for the agent
AGENT_HUMAN_TEMPLATE = """Below are examples of similar conversation states and their responses:

{examples}

---

{goal_line}

Current conversation:
{current_conversation}

Based on the examples, and the current conversation, answer the following questions:
1. Would a real agent respond next?
2. If so, what would the response be?
3. If not, why not?

Response format directions:
{format_instructions}
"""

# Create the prompt with both system and human messages
AGENT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=AGENT_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(AGENT_HUMAN_TEMPLATE)
])
