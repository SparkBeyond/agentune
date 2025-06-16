import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from conversation_simulator.rag.commons import _format_conversation_history
from conversation_simulator.participants.customer.rag.rag import RagCustomer

# Import common test utilities
from tests.rag.common_test_utils import (
    FS_MOCK_CUSTOMER_DOC_1,
    FS_MOCK_CUSTOMER_DOC_2,
    Message,
    ParticipantRole,
)


# --- Unit Tests for get_few_shot_examples_for_customer ---

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_customer_success():
    mock_customer_store = MagicMock(spec=VectorStore)
    mock_customer_store.index = MagicMock()
    mock_customer_store.index.ntotal = 3 # Simulate a non-empty store

    mock_history = [
        Message(sender=ParticipantRole.AGENT, content="How can I help you today?", timestamp=datetime.now())
    ]
    formatted_history_query = _format_conversation_history(mock_history)

    # Simulate asimilarity_search returning relevant customer documents
    retrieved_docs_from_search = [
        Document(page_content=FS_MOCK_CUSTOMER_DOC_1["page_content"], metadata=FS_MOCK_CUSTOMER_DOC_1["metadata"]),
        Document(page_content=FS_MOCK_CUSTOMER_DOC_2["page_content"], metadata=FS_MOCK_CUSTOMER_DOC_2["metadata"]),
    ]
    mock_customer_store.asimilarity_search = AsyncMock(return_value=retrieved_docs_from_search)

    examples = await RagCustomer._get_few_shot_examples(
        conversation_history=mock_history,
        vector_store=mock_customer_store,
        k=2
    )

    assert len(examples) == 2
    assert isinstance(examples[0], Document)
    
    # Check content of the first returned Document
    assert examples[0].page_content == FS_MOCK_CUSTOMER_DOC_1["page_content"]
    assert examples[0].metadata["content"] == FS_MOCK_CUSTOMER_DOC_1["metadata"]["content"]
    assert examples[0].metadata["role"] == ParticipantRole.CUSTOMER.value

    # Check content of the second returned Document
    assert examples[1].page_content == FS_MOCK_CUSTOMER_DOC_2["page_content"]
    assert examples[1].metadata["content"] == FS_MOCK_CUSTOMER_DOC_2["metadata"]["content"]
    assert examples[1].metadata["role"] == ParticipantRole.CUSTOMER.value
    
    mock_customer_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=2)

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_customer_empty_store():
    mock_customer_store_empty = MagicMock(spec=VectorStore)
    mock_customer_store_empty.index = MagicMock()
    mock_customer_store_empty.index.ntotal = 0  # Empty store
    mock_customer_store_empty.asimilarity_search = AsyncMock(return_value=[])  # Empty result

    mock_history = [Message(sender=ParticipantRole.AGENT, content="Any query for empty store", timestamp=datetime.now())]

    # Test with empty store
    with pytest.raises(ValueError, match="No documents retrieved from vector store"):
        await RagCustomer._get_few_shot_examples(
            conversation_history=mock_history,
            vector_store=mock_customer_store_empty,
            k=2
        )

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_customer_no_docs_retrieved():
    mock_customer_store = MagicMock(spec=VectorStore)
    mock_customer_store.index = MagicMock()
    mock_customer_store.index.ntotal = 1  # Simulate non-empty store
    mock_customer_store.asimilarity_search = AsyncMock(return_value=[])  # Search returns empty list

    mock_history = [Message(sender=ParticipantRole.AGENT, content="A query, no docs", timestamp=datetime.now())]
    formatted_history_query = _format_conversation_history(mock_history)

    with pytest.raises(ValueError, match="No documents retrieved from vector store"):
        await RagCustomer._get_few_shot_examples(
            conversation_history=mock_history,
            vector_store=mock_customer_store,
            k=2
        )
    mock_customer_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=2)

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_customer_less_than_k_valid():
    """Test when fewer valid documents are found than requested (k)."""
    mock_customer_store = MagicMock(spec=VectorStore)
    mock_customer_store.index = MagicMock()
    mock_customer_store.index.ntotal = 1  # Only 1 document in store
    
    # Only return 1 document when 2 were requested
    retrieved_docs = [
        Document(page_content=FS_MOCK_CUSTOMER_DOC_1["page_content"], metadata=FS_MOCK_CUSTOMER_DOC_1["metadata"])
    ]
    mock_customer_store.asimilarity_search = AsyncMock(return_value=retrieved_docs)

    mock_history = [Message(sender=ParticipantRole.AGENT, content="Query with limited results", timestamp=datetime.now())]
    formatted_history_query = _format_conversation_history(mock_history)

    # Should raise an error when we can't get enough valid documents
    with pytest.raises(ValueError, match="Not enough valid documents retrieved from vector store"):
        await RagCustomer._get_few_shot_examples(
            conversation_history=mock_history,
            vector_store=mock_customer_store,
            k=2  # Requesting 2 but only 1 is available
        )
    
    mock_customer_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=2)

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_customer_missing_metadata():
    """Test handling of documents with missing required metadata."""
    mock_customer_store = MagicMock(spec=VectorStore)
    mock_customer_store.index = MagicMock()
    mock_customer_store.index.ntotal = 2  # Simulate non-empty store
    
    # Document with missing required metadata
    invalid_doc = Document(
        page_content="Some content",
        metadata={"conversation_id": "test_conv", "role": "customer"}  # Missing timestamp
    )
    mock_customer_store.asimilarity_search = AsyncMock(return_value=[invalid_doc])

    mock_history = [Message(sender=ParticipantRole.AGENT, content="Query with invalid metadata", timestamp=datetime.now())]
    
    # Should raise an error when no valid documents are found
    with pytest.raises(ValueError, match="Not enough valid documents retrieved from vector store"):
        await RagCustomer._get_few_shot_examples(
            conversation_history=mock_history,
            vector_store=mock_customer_store,
            k=1
        )

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_customer_wrong_role_in_metadata():
    """Test filtering of documents with incorrect role in metadata."""
    mock_customer_store = MagicMock(spec=VectorStore)
    mock_customer_store.index = MagicMock()
    mock_customer_store.index.ntotal = 2  # Simulate non-empty store
    
    # Document with wrong role (should be CUSTOMER)
    wrong_role_doc = Document(
        page_content="Some content",
        metadata={
            "conversation_id": "test_conv",
            "role": "agent",  # Wrong role
            "timestamp": datetime.now().isoformat(),
            "content": "Some content"
        }
    )
    mock_customer_store.asimilarity_search = AsyncMock(return_value=[wrong_role_doc])

    mock_history = [Message(sender=ParticipantRole.AGENT, content="Query with wrong role doc", timestamp=datetime.now())]
    
    # Should raise an error when no valid documents are found
    with pytest.raises(ValueError, match="Not enough valid documents retrieved from vector store"):
        await RagCustomer._get_few_shot_examples(
            conversation_history=mock_history,
            vector_store=mock_customer_store,
            k=1
        )

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_customer_similarity_search_error():
    """Test error handling during similarity search."""
    mock_customer_store = MagicMock(spec=VectorStore)
    mock_customer_store.index = MagicMock()
    mock_customer_store.index.ntotal = 1  # Simulate non-empty store
    mock_customer_store.asimilarity_search = AsyncMock(side_effect=Exception("Customer DB error"))

    mock_history = [Message(sender=ParticipantRole.AGENT, content="Query causing customer DB error", timestamp=datetime.now())]
    formatted_history_query = _format_conversation_history(mock_history)

    with pytest.raises(Exception, match="Customer DB error"):
        await RagCustomer._get_few_shot_examples(
            conversation_history=mock_history,
            vector_store=mock_customer_store,
            k=2
        )
    mock_customer_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=2)
