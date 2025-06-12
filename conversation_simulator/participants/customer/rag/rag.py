"""RAG-based customer participant implementation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple, cast

from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ....models import Conversation, Message, ParticipantRole
from ..base import Customer
from ....rag import get_few_shot_examples_for_customer
from ....rag.processing import convert_message_to_langchain


CUSTOMER_SYSTEM_PROMPT = """You are a customer interacting with a customer service agent.
- Your goal is to resolve your issue.
- Behave like a real person. You can be frustrated, confused, or happy depending on the agent's responses.
- Use the provided few-shot examples to understand the tone and style of a typical customer in this situation.
- Use the conversation history to stay in context.
- Keep your responses concise and to the point.
"""


class RagCustomer(Customer):
    """RAG LLM-based customer participant."""

    llm_chain: Runnable

    def __init__(
        self,
        customer_vector_store: FAISS,
        agent_vector_store: FAISS,
        all_conversations: List[Dict[str, Any]],
        openai_api_key: str,
    ):
        super().__init__()
        self.customer_vector_store = customer_vector_store
        self.agent_vector_store = agent_vector_store
        self.all_conversations = all_conversations
        self.openai_api_key = openai_api_key
        self.intent_description: str | None = None
        self.llm_chain = self._create_llm_chain()

    def _create_llm_chain(self, intent_description: str | None = None) -> Runnable:
        """Creates the LangChain Expression Language (LCEL) chain for the customer."""
        base_system_prompt = CUSTOMER_SYSTEM_PROMPT

        if intent_description:
            system_prompt_content = f"Your goal: {intent_description}\n\n{base_system_prompt}"
        elif self.intent_description:
            system_prompt_content = f"Your goal: {self.intent_description}\n\n{base_system_prompt}"
        else:
            system_prompt_content = base_system_prompt

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt_content),
                MessagesPlaceholder(variable_name="few_shot_examples"),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{last_agent_message}"),
            ]
        )
        llm = ChatOpenAI(api_key=SecretStr(self.openai_api_key), model="gpt-4o-mini")
        return prompt | llm | StrOutputParser()

    def with_intent(self, intent_description: str) -> RagCustomer:
        """Return a new RagCustomer instance with the specified intent."""
        new_customer = RagCustomer(
            customer_vector_store=self.customer_vector_store,
            agent_vector_store=self.agent_vector_store,
            all_conversations=self.all_conversations,
            openai_api_key=self.openai_api_key,
        )
        new_customer.intent_description = intent_description
        new_customer.llm_chain = new_customer._create_llm_chain(intent_description=intent_description)
        return new_customer

    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate next customer message using RAG LLM approach."""
        if not conversation.messages or conversation.messages[-1].sender != ParticipantRole.AGENT:
            return None

        last_message = conversation.messages[-1]

        # 1. Retrieval
        few_shot_examples = await get_few_shot_examples_for_customer(
            agent_message=last_message,
            customer_vector_store=self.customer_vector_store,
            all_conversations=self.all_conversations,
        )

        # 2. Augmentation
        formatted_examples = self._format_examples(few_shot_examples)
        chat_history_langchain = cast(List[BaseMessage], [
            convert_message_to_langchain(msg) for msg in conversation.messages[:-1]
        ])

        # 3. Generation
        response_content = await self.llm_chain.ainvoke(
            {
                "few_shot_examples": formatted_examples,
                "chat_history": chat_history_langchain,
                "last_agent_message": last_message.content,
            }
        )

        # 4. Return Message
        return Message(
            sender=self.role,
            content=response_content,
            timestamp=datetime.now(),
        )

    def _format_examples(
        self, examples: List[Tuple[Dict, Dict]]
    ) -> List[BaseMessage]:
        """Formats the retrieved few-shot examples into a list of LangChain messages."""
        messages: List[BaseMessage] = []
        for agent_msg, customer_msg in examples:
            messages.append(HumanMessage(content=agent_msg["content"]))  # Agent's historical turn
            messages.append(AIMessage(content=customer_msg["content"])) # Customer's historical turn
        return messages
