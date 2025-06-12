"""RAG-based agent participant implementation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ....models import Conversation, Message, ParticipantRole
from ..base import Agent
from ....rag.processing import convert_message_to_langchain
from ....rag import get_few_shot_examples_for_agent


class RagAgent(Agent):
    """RAG LLM-based agent participant.
    
    Uses a language model with Retrieval Augmented Generation
    to generate agent responses.
    """

    def __init__(
        self,
        agent_vector_store: FAISS,
        customer_vector_store: FAISS,
        all_conversations: List[Dict[str, Any]],
        openai_api_key: str,
    ):
        """Initializes the RAG agent.

        Args:
            agent_vector_store: FAISS vector store for agent messages.
            customer_vector_store: FAISS vector store for customer messages.
            all_conversations: The full dataset of conversations for context lookup.
        """
        super().__init__()
        self.agent_vector_store = agent_vector_store
        self.customer_vector_store = customer_vector_store
        self.all_conversations = all_conversations
        self.openai_api_key = openai_api_key
        self.intent_description: str | None = None # Store intent
        self.llm_chain = self._create_llm_chain()
    
    def _create_llm_chain(self, intent_description: str | None = None) -> Runnable:
        """Creates the LangChain Expression Language (LCEL) chain for the agent."""
        # Construct the system prompt
        base_system_prompt = """You are a helpful customer service agent.
- Your primary goal is to assist the user with their issue based on the conversation history and relevant examples.
- If few-shot examples are provided, use them to understand the tone, style, and common solutions.
- If no examples are relevant, rely on your general knowledge and the conversation history.
- Keep your responses clear, concise, and professional."""

        current_intent = intent_description or self.intent_description
        if current_intent:
            system_prompt_content = f"Your primary goal is: {current_intent}\n\n{base_system_prompt}"
        else:
            system_prompt_content = base_system_prompt

        # The `MessagesPlaceholder` is a special variable that can hold a sequence of messages.
        # `few_shot_examples` will be a list of alternating Human/AI messages.
        # `chat_history` will be the messages from the current conversation.
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt_content),
            MessagesPlaceholder(variable_name="few_shot_examples"),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{last_customer_message}"),
        ])

        llm = ChatOpenAI(api_key=SecretStr(self.openai_api_key), model="gpt-4o-mini")

        # Return the runnable chain
        return prompt_template | llm | StrOutputParser()

    def with_intent(self, intent_description: str) -> RagAgent:
        """Return a new RagAgent instance with the specified intent.
        
        The intent is primarily used to guide the initial system prompt.
        """
        new_agent = RagAgent(
            agent_vector_store=self.agent_vector_store,
            customer_vector_store=self.customer_vector_store,
            all_conversations=self.all_conversations,
            openai_api_key=self.openai_api_key,
        )
        new_agent.intent_description = intent_description
        # Re-create the LLM chain with the new intent description
        new_agent.llm_chain = new_agent._create_llm_chain(intent_description=intent_description)
        return new_agent

    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate next agent message using RAG LLM approach."""
        if not conversation.messages or conversation.messages[-1].sender != ParticipantRole.CUSTOMER:
            return None

        last_message = conversation.messages[-1]

        # 1. Retrieval
        few_shot_examples = await get_few_shot_examples_for_agent(
            last_customer_message_content=last_message.content,
            agent_vector_store=self.agent_vector_store,
            all_conversations=self.all_conversations,
            k=3,
        )

        # 2. Augmentation
        # Format few-shot examples and history for the prompt template
        formatted_examples = self._format_examples(few_shot_examples)
        chat_history = [convert_message_to_langchain(msg) for msg in conversation.messages[:-1]]

        # 3. Generation
        response_content = await self.llm_chain.ainvoke({
            "few_shot_examples": formatted_examples,
            "chat_history": chat_history,
            "last_customer_message": last_message.content,
        })

        # 4. Return Message
        return Message(
            sender=self.role, content=response_content, timestamp=datetime.now()
        )

    def _format_examples(self, examples: List[Tuple[Dict, Dict]]) -> List[BaseMessage]:
        """Formats the retrieved few-shot examples into a list of LangChain messages."""
        messages: List[BaseMessage] = []
        for user_example, assistant_example in examples:
            messages.extend([
                HumanMessagePromptTemplate.from_template(user_example["content"]).format(),
                AIMessage(content=assistant_example["content"]),
            ])
        return messages
