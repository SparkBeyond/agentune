"""Prompts for the RAG customer participant."""


from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# System prompt for the customer
CUSTOMER_SYSTEM_PROMPT = """You are simulating a customer in a text-based customer service conversation. 

Pay close attention to the tone and emotion in the example responses and mimic the conversations in the best way possible.
Note that customers are prone to not answering, so if you think a customer would not respond, set should_respond to false.

Your response MUST be a JSON object with the following structure, written from the customer's perspective:
{
    "reasoning": "My reasoning for my next action as the customer is...",
    "should_respond": true/false,
    "response": "My response as the customer is... (or null)" 
}
"""

# Human message template for the customer
CUSTOMER_HUMAN_TEMPLATE = """Below are examples of similar full conversations and the responses of the customer for the similar stage:

{examples}

---

Additional context:
{goal_line}

---

Current conversation:
{current_conversation}
"""

# Create the prompt with both system and human messages
CUSTOMER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=CUSTOMER_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(CUSTOMER_HUMAN_TEMPLATE)
])