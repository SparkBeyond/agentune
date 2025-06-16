"""Integration tests for RAG processing functionality."""

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from datetime import datetime
import os
from langchain_openai import OpenAIEmbeddings

from conversation_simulator.models import Conversation, Message, ParticipantRole
from conversation_simulator.rag import conversations_to_langchain_documents


# Mock conversation data for integration tests
MOCK_INTEGRATION_CONVERSATIONS = [
    Conversation(
        messages=tuple([
            Message(sender=ParticipantRole.CUSTOMER, content="This is a customer query for integration testing.", timestamp=datetime.now()),
            Message(sender=ParticipantRole.AGENT, content="This is an agent response for integration testing.", timestamp=datetime.now()),
            Message(sender=ParticipantRole.CUSTOMER, content="Follow-up customer message.", timestamp=datetime.now()),
        ])
    ),
    Conversation(
        messages=tuple([
            Message(sender=ParticipantRole.CUSTOMER, content="Only customer messages here.", timestamp=datetime.now()),
        ])
    )
]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_vector_stores_integration():
    """
    Tests create_vector_stores_from_conversations with real OpenAI API calls,
    creating in-memory vector stores.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY not set, skipping RAG integration test.")

    customer_store: VectorStore | None = None
    agent_store: VectorStore | None = None

    customer_documents = conversations_to_langchain_documents(MOCK_INTEGRATION_CONVERSATIONS, ParticipantRole.CUSTOMER)
    agent_documents = conversations_to_langchain_documents(MOCK_INTEGRATION_CONVERSATIONS, ParticipantRole.AGENT)
    customer_store = await FAISS.afrom_documents(customer_documents, OpenAIEmbeddings(model="text-embedding-ada-002"))
    agent_store = await FAISS.afrom_documents(agent_documents, OpenAIEmbeddings(model="text-embedding-ada-002"))
    
    assert customer_store is not None, "Customer store creation failed"
    assert agent_store is not None, "Agent store creation failed"

    # Assertions
    assert isinstance(customer_store, VectorStore), "Customer store is not a VectorStore instance."
    assert isinstance(agent_store, VectorStore), "Agent store is not a VectorStore instance."

    # Check if customer_store has content by performing a search
    customer_results = await customer_store.asimilarity_search("customer query", k=1)
    # MOCK_INTEGRATION_CONVERSATIONS has customer messages, so we expect results
    assert len(customer_results) > 0, "Customer store similarity search returned no results when it should have."
    assert isinstance(customer_results[0], Document), "Customer search result is not a Document."
    print(f"Customer search result: {customer_results[0].page_content}")

    # Check if agent_store has content by performing a search
    agent_results = await agent_store.asimilarity_search("agent response", k=1)
    # MOCK_INTEGRATION_CONVERSATIONS has agent messages, so we expect results
    assert len(agent_results) > 0, "Agent store similarity search returned no results when it should have."
    assert isinstance(agent_results[0], Document), "Agent search result is not a Document."
    print(f"Agent search result: {agent_results[0].page_content}")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario_name, conversation_input, description_suffix",
    [
        (
            "single_customer_message",
            [Conversation(messages=tuple([Message(sender=ParticipantRole.CUSTOMER, content="hello", timestamp=datetime.now())]))],
            "single-customer-message conversation"
        ),
        (
            "single_agent_message",
            [Conversation(messages=tuple([Message(sender=ParticipantRole.AGENT, content="world", timestamp=datetime.now())]))],
            "single-agent-message conversation"
        ),
        (
            "no_messages",
            [Conversation(messages=tuple())],
            "zero-message conversation"
        ),
    ]
)
async def test_empty_conversations_integration(
    caplog,
    scenario_name: str,
    conversation_input: list[Conversation],
    description_suffix: str
):
    """
    Tests that FAISS.afrom_documents raises IndexError for empty document lists,
    which occurs with conversations that don't yield valid documents (e.g., < 2 messages).
    Covers different types of sparse conversations.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY not set, skipping RAG integration test.")

    # Generate documents for customer and agent roles
    customer_docs = conversations_to_langchain_documents(conversation_input, ParticipantRole.CUSTOMER)
    agent_docs = conversations_to_langchain_documents(conversation_input, ParticipantRole.AGENT)

    # Assert that document lists are empty for these scenarios
    assert not customer_docs, f"Customer docs should be empty for {description_suffix} ({scenario_name})"
    assert not agent_docs, f"Agent docs should be empty for {description_suffix} ({scenario_name})"

    # Assert FAISS raises IndexError when trying to create a store from empty documents
    with pytest.raises(IndexError):
        await FAISS.afrom_documents(customer_docs, OpenAIEmbeddings(model="text-embedding-ada-002"))
    
    with pytest.raises(IndexError):
        await FAISS.afrom_documents(agent_docs, OpenAIEmbeddings(model="text-embedding-ada-002"))
    
    caplog.clear()
