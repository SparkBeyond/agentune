"""Integration tests for RAG processing functionality."""

import pytest
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore
from langchain_core.documents import Document
from datetime import datetime
import os
from langchain_openai import OpenAIEmbeddings

from agentune.simulate.models import Conversation, Message, ParticipantRole
from agentune.simulate.rag import conversations_to_langchain_documents
from agentune.simulate.rag.filtered_retriever import VectorStoreSearcher


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

    # Convert conversations to documents
    documents = conversations_to_langchain_documents(MOCK_INTEGRATION_CONVERSATIONS)
    
    # Create a single vector store
    vector_store = InMemoryVectorStore(embedding=OpenAIEmbeddings(model="text-embedding-ada-002"))
    await vector_store.aadd_documents(documents)
    
    assert vector_store is not None, "Vector store creation failed"

    # Assertions
    assert isinstance(vector_store, VectorStore), "Vector store is not a VectorStore instance."

    # Check if vector_store has content by performing customer-related search using VectorStoreSearcher
    searcher = VectorStoreSearcher.create(vector_store)
    customer_results_with_scores = await searcher.similarity_search_with_filter(
        query="customer query",
        k=1,
        filter_dict={"next_message_role": ParticipantRole.CUSTOMER.value}
    )
    # MOCK_INTEGRATION_CONVERSATIONS has customer messages, so we expect results
    assert len(customer_results_with_scores) > 0, "Vector store similarity search for customer messages returned no results when it should have."
    customer_doc, customer_score = customer_results_with_scores[0]
    assert isinstance(customer_doc, Document), "Customer search result is not a Document."
    assert isinstance(customer_score, float), "Customer search score is not a float."
    print(f"Customer search result: {customer_doc.page_content} (score: {customer_score})")

    # Check if vector_store has content by performing agent-related search using VectorStoreSearcher
    agent_results_with_scores = await searcher.similarity_search_with_filter(
        query="agent response",
        k=1,
        filter_dict={"next_message_role": ParticipantRole.AGENT.value}
    )
    # MOCK_INTEGRATION_CONVERSATIONS has agent messages, so we expect results
    assert len(agent_results_with_scores) > 0, "Vector store similarity search for agent messages returned no results when it should have."
    agent_doc, agent_score = agent_results_with_scores[0]
    assert isinstance(agent_doc, Document), "Agent search result is not a Document."
    assert isinstance(agent_score, float), "Agent search score is not a float."
    print(f"Agent search result: {agent_doc.page_content} (score: {agent_score})")
