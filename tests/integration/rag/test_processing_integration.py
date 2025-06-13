
import pytest
from langchain_community.vectorstores import FAISS # Still needed for loading/saving functionality
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document # For creating dummy docs if needed
from datetime import datetime

from conversation_simulator.models import Conversation, Message, ParticipantRole
from conversation_simulator.rag import create_vector_stores_from_conversations



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

@pytest.mark.asyncio
async def test_create_vector_stores_integration():
    """
    Tests create_vector_stores_from_conversations with real OpenAI API calls,
    creating in-memory vector stores.
    """
    # 1. Initialize OpenAIEmbeddings (implicitly handled by create_vector_stores_from_conversations)
    # The openai_api_key fixture from conftest.py provides the key

    customer_store: VectorStore | None = None
    agent_store: VectorStore | None = None

    # 2. Create vector stores
    print("Creating vector stores using OpenAI API...")
    # We explicitly pass FAISS here because the underlying create_vector_stores_from_conversations
    # defaults to FAISS. If that default changes, this test might need adjustment
    # or we make vector_store_class mandatory in the main function.
    customer_store, agent_store = await create_vector_stores_from_conversations(
        conversations=MOCK_INTEGRATION_CONVERSATIONS,
        openai_embedding_model_name="text-embedding-ada-002",
        vector_store_class=FAISS # Explicitly use FAISS for this test's purpose if needed, or remove if truly generic
    )
    
    assert customer_store is not None, "Customer store creation failed"
    assert agent_store is not None, "Agent store creation failed"

    # 3. Assertions
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


@pytest.mark.asyncio
async def test_empty_conversations_integration(caplog):
    """
    Tests that create_vector_stores_from_conversations handles empty or no relevant conversations
    by creating empty FAISS stores.
    """
    # Test with completely empty conversations list
    customer_store, agent_store = await create_vector_stores_from_conversations(
        conversations=[],
    )
    assert "No conversations provided. Returning empty vector stores." in caplog.text
    assert isinstance(customer_store, VectorStore)
    assert isinstance(agent_store, VectorStore)
    customer_results = await customer_store.asimilarity_search("any query", k=1)
    assert len(customer_results) == 1 and customer_results[0].page_content == "dummy", "Customer store should contain only a dummy document"
    agent_results = await agent_store.asimilarity_search("any query", k=1)
    assert len(agent_results) == 1 and agent_results[0].page_content == "dummy", "Agent store should contain only a dummy document"
    caplog.clear()

    # Test with conversations that have no agent messages
    no_agent_messages_conv = [
        Conversation(messages=tuple([
            Message(sender=ParticipantRole.CUSTOMER, content="hello", timestamp=datetime.now())
        ]))
    ]
    customer_store, agent_store = await create_vector_stores_from_conversations(
        conversations=no_agent_messages_conv
    )
    assert isinstance(customer_store, VectorStore)
    assert isinstance(agent_store, VectorStore)
    customer_results = await customer_store.asimilarity_search("any query", k=1)
    assert len(customer_results) == 1 and customer_results[0].page_content == "dummy", "Customer store should be dummy for single-message conv"
    agent_results = await agent_store.asimilarity_search("any query", k=1)
    assert len(agent_results) == 1 and agent_results[0].page_content == "dummy", "Agent store should be dummy for single-message conv"
    caplog.clear()

    # Test with conversations that have no customer messages
    no_customer_messages_conv = [
        Conversation(messages=tuple([
            Message(sender=ParticipantRole.AGENT, content="world", timestamp=datetime.now())
        ]))
    ]
    customer_store, agent_store = await create_vector_stores_from_conversations(
        conversations=no_customer_messages_conv
    )
    assert isinstance(customer_store, VectorStore)
    assert isinstance(agent_store, VectorStore)
    customer_results = await customer_store.asimilarity_search("any query", k=1)
    assert len(customer_results) == 1 and customer_results[0].page_content == "dummy", "Customer store should be dummy for single-message conv"
    agent_results = await agent_store.asimilarity_search("any query", k=1)
    assert len(agent_results) == 1 and agent_results[0].page_content == "dummy", "Agent store should be dummy for single-message conv"
    caplog.clear()

    # Test with conversations that have no messages at all (empty message list in Conversation object)
    all_empty_messages_conv = [
        Conversation(messages=tuple())
    ]
    customer_store, agent_store = await create_vector_stores_from_conversations(
        conversations=all_empty_messages_conv
    )
    assert isinstance(customer_store, VectorStore)
    assert isinstance(agent_store, VectorStore)
    customer_results = await customer_store.asimilarity_search("any query", k=1)
    assert len(customer_results) == 1 and customer_results[0].page_content == "dummy", "Customer store should contain only a dummy document"
    agent_results = await agent_store.asimilarity_search("any query", k=1)
    assert len(agent_results) == 1 and agent_results[0].page_content == "dummy", "Agent store should contain only a dummy document"
    caplog.clear()
