import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from conversation_simulator.rag.indexing_and_retrieval_utils import _format_conversation_history
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
    # Create documents with the expected next_message_content field
    doc1_metadata = FS_MOCK_CUSTOMER_DOC_1["metadata"].copy()
    doc1_metadata["next_message_content"] = doc1_metadata.get("content", "")
    
    doc2_metadata = FS_MOCK_CUSTOMER_DOC_2["metadata"].copy()
    doc2_metadata["next_message_content"] = doc2_metadata.get("content", "")
    
    retrieved_docs_from_search = [
        Document(page_content=FS_MOCK_CUSTOMER_DOC_1["page_content"], metadata=doc1_metadata),
        Document(page_content=FS_MOCK_CUSTOMER_DOC_2["page_content"], metadata=doc2_metadata),
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
    
    mock_customer_store.asimilarity_search.assert_called_once_with(
        query=formatted_history_query, 
        k=2,
        filter={"next_message_role": ParticipantRole.CUSTOMER.value}
    )

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_customer_empty_store():
    mock_customer_store_empty = MagicMock(spec=VectorStore)
    mock_customer_store_empty.index = MagicMock()
    mock_customer_store_empty.index.ntotal = 0  # Empty store
    mock_customer_store_empty.asimilarity_search = AsyncMock(return_value=[])  # Empty result

    mock_history = [Message(sender=ParticipantRole.AGENT, content="Any query for empty store", timestamp=datetime.now())]

    # Test with empty store
    with pytest.raises(ValueError, match="Not enough valid documents retrieved for role customer"):
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

    with pytest.raises(ValueError, match="Not enough valid documents retrieved for role customer"):
        await RagCustomer._get_few_shot_examples(
            conversation_history=mock_history,
            vector_store=mock_customer_store,
            k=2
        )
    mock_customer_store.asimilarity_search.assert_called_once_with(
        query=formatted_history_query, 
        k=2,
        filter={"next_message_role": ParticipantRole.CUSTOMER.value}
    )

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
    with pytest.raises(ValueError, match="Not enough valid documents retrieved for role customer"):
        await RagCustomer._get_few_shot_examples(
            conversation_history=mock_history,
            vector_store=mock_customer_store,
            k=2  # Requesting 2 but only 1 is available
        )
    
    mock_customer_store.asimilarity_search.assert_called_once_with(
        query=formatted_history_query, 
        k=2,
        filter={"next_message_role": ParticipantRole.CUSTOMER.value}
    )

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_customer_new_approach():
    mock_customer_store = MagicMock(spec=VectorStore)
    mock_customer_store.index = MagicMock()
    mock_customer_store.index.ntotal = 1  # Simulate non-empty store
    
    # Create documents with proper metadata structure needed by the new implementation
    doc1 = Document(
        page_content="Customer: I need help with my account setup.",
        metadata={
            "conversation_id": "test_conv_1",
            "next_message_role": ParticipantRole.CUSTOMER.value,
            "next_message_content": "I need help with my account setup."
        }
    )
    doc2 = Document(
        page_content="Customer: Where can I find my order history?",
        metadata={
            "conversation_id": "test_conv_2",
            "next_message_role": ParticipantRole.CUSTOMER.value,
            "next_message_content": "Where can I find my order history?"
        }
    )
    
    mock_customer_store.asimilarity_search = AsyncMock(return_value=[doc1, doc2])

    mock_history = [Message(sender=ParticipantRole.AGENT, content="A query, no docs", timestamp=datetime.now())]
    formatted_history_query = _format_conversation_history(mock_history)

    # Execute function under test
    result = await RagCustomer._get_few_shot_examples(
        conversation_history=mock_history,
        vector_store=mock_customer_store,
        k=2
    )

    # Verify - note the filter parameter is now expected
    mock_customer_store.asimilarity_search.assert_called_once_with(
        query=formatted_history_query, 
        k=2,
        filter={"next_message_role": ParticipantRole.CUSTOMER.value}
    )
    assert len(result) == 2
    assert isinstance(result[0], Document)
    assert result[0].metadata["next_message_content"] == "I need help with my account setup."
    assert result[1].metadata["next_message_content"] == "Where can I find my order history?"

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_customer_no_results():
    mock_customer_store = MagicMock(spec=VectorStore)
    mock_customer_store.index = MagicMock()
    mock_customer_store.index.ntotal = 5  # Non-empty store

    mock_history = [Message(sender=ParticipantRole.AGENT, content="Query with no results", timestamp=datetime.now())]
    formatted_history_query = _format_conversation_history(mock_history)
    
    # Setup mock to return empty list
    mock_customer_store.asimilarity_search = AsyncMock(return_value=[])
    
    # Execute function and verify exception
    with pytest.raises(ValueError, match="Not enough valid documents retrieved for role customer"):
        await RagCustomer._get_few_shot_examples(
            conversation_history=mock_history,
            vector_store=mock_customer_store,
            k=2
        )
    
    # Verify mock was called with filter
    mock_customer_store.asimilarity_search.assert_called_once_with(
        query=formatted_history_query, 
        k=2,
        filter={"next_message_role": ParticipantRole.CUSTOMER.value}
    )

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
    with pytest.raises(ValueError, match="Not enough valid documents retrieved for role customer"):
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
    mock_customer_store.asimilarity_search.assert_called_once_with(
        query=formatted_history_query, 
        k=2, 
        filter={"next_message_role": ParticipantRole.CUSTOMER.value}
    )
