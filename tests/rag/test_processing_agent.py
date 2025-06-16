import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from conversation_simulator.rag.commons import _format_conversation_history
from conversation_simulator.participants.agent.rag.rag import RagAgent

# Import common test utilities
from tests.rag.common_test_utils import (
    FS_MOCK_AGENT_DOC_1,
    FS_MOCK_AGENT_DOC_4_FOLLOWUP,
    Message,
    ParticipantRole,
    create_mock_doc_data,
    TIMESTAMP_STR_MINUS_5S,
    TIMESTAMP_MINUS_10S,
)


# --- Unit Tests for get_few_shot_examples_for_agent ---


@pytest.mark.asyncio
async def test_get_few_shot_examples_for_agent_success():
    mock_agent_store = MagicMock(spec=VectorStore)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 3 # Simulate a non-empty store

    mock_history = [Message(sender=ParticipantRole.CUSTOMER, content="Tell me about product X", timestamp=datetime.now())]
    formatted_history_query = _format_conversation_history(mock_history)

    # Simulate asimilarity_search returning the 2 relevant agent documents for k=2.
    retrieved_docs = [
        Document(page_content=FS_MOCK_AGENT_DOC_1["page_content"], metadata=FS_MOCK_AGENT_DOC_1["metadata"]),
        Document(page_content=FS_MOCK_AGENT_DOC_4_FOLLOWUP["page_content"], metadata=FS_MOCK_AGENT_DOC_4_FOLLOWUP["metadata"]),
    ]
    mock_agent_store.asimilarity_search = AsyncMock(return_value=retrieved_docs)

    examples = await RagAgent._get_few_shot_examples(
        conversation_history=mock_history,
        vector_store=mock_agent_store,
        k=2
    )

    assert len(examples) == 2
    assert isinstance(examples[0], Document)
    assert examples[0].metadata["role"] == ParticipantRole.AGENT.value
    assert examples[0].metadata["content"] == FS_MOCK_AGENT_DOC_1["metadata"]["content"]
    assert examples[1].metadata["content"] == FS_MOCK_AGENT_DOC_4_FOLLOWUP["metadata"]["content"]
    
    mock_agent_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=2)

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_agent_empty_store():
    mock_agent_store_empty = MagicMock(spec=VectorStore)
    mock_agent_store_empty.index = MagicMock()
    mock_agent_store_empty.index.ntotal = 0  # Empty store
    mock_agent_store_empty.asimilarity_search = AsyncMock(return_value=[])

    mock_history = [Message(sender=ParticipantRole.CUSTOMER, content="Any query", timestamp=datetime.now())]

    # Test with empty store
    with pytest.raises(ValueError, match="No documents retrieved from vector store"):
        await RagAgent._get_few_shot_examples(
            conversation_history=mock_history,
            vector_store=mock_agent_store_empty,
            k=2
        )

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_agent_no_docs_retrieved():
    mock_agent_store = MagicMock(spec=VectorStore)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 1
    mock_agent_store.asimilarity_search = AsyncMock(return_value=[])  # Search returns empty

    mock_history = [Message(sender=ParticipantRole.CUSTOMER, content="A query", timestamp=datetime.now())]
    formatted_history_query = _format_conversation_history(mock_history)

    with pytest.raises(ValueError, match="No documents retrieved from vector store"):
        await RagAgent._get_few_shot_examples(
            conversation_history=mock_history,
            vector_store=mock_agent_store,
            k=2
        )
    mock_agent_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=2)

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_agent_less_than_k_valid():
    mock_agent_store = MagicMock(spec=VectorStore)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 1

    mock_history = [Message(sender=ParticipantRole.CUSTOMER, content="Another query", timestamp=datetime.now())]
    formatted_history_query = _format_conversation_history(mock_history)

    # One doc is valid, the other is missing a timestamp in metadata.
    doc_with_ts = Document(page_content=FS_MOCK_AGENT_DOC_1["page_content"], metadata=FS_MOCK_AGENT_DOC_1["metadata"])
    
    # Create a mock document specifically missing the 'timestamp' in its metadata
    mock_doc_data_no_ts = create_mock_doc_data(
        history_messages=[Message(sender=ParticipantRole.CUSTOMER, content="History for no ts doc", timestamp=TIMESTAMP_MINUS_10S)],
        next_message_role=ParticipantRole.AGENT.value,
        next_message_content="Agent content for no ts doc",
        next_message_timestamp_str=TIMESTAMP_STR_MINUS_5S, # This will be removed
        conversation_id="conv_agent_no_ts", message_index=1,
        remove_keys=["timestamp"]
    )
    doc_without_ts = Document(page_content=mock_doc_data_no_ts["page_content"], metadata=mock_doc_data_no_ts["metadata"])
    retrieved_docs = [doc_with_ts, doc_without_ts]
    mock_agent_store.asimilarity_search = AsyncMock(return_value=retrieved_docs)

    with pytest.raises(ValueError, match="Not enough valid documents retrieved from vector store"):
        await RagAgent._get_few_shot_examples(
            conversation_history=mock_history,
            vector_store=mock_agent_store,
            k=3
        )
    mock_agent_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=3)

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_agent_missing_metadata():
    mock_agent_store = MagicMock(spec=VectorStore)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 2  # Simulate a non-empty store

    mock_history = [Message(sender=ParticipantRole.CUSTOMER, content="Query for missing metadata", timestamp=datetime.now())]
    formatted_history_query = _format_conversation_history(mock_history)

    # Create a document missing 'content' in its metadata
    mock_doc_data_missing_content = create_mock_doc_data(
        history_messages=[Message(sender=ParticipantRole.CUSTOMER, content="History for missing content doc", timestamp=TIMESTAMP_MINUS_10S)],
        next_message_role=ParticipantRole.AGENT.value,
        next_message_content="This content will be removed", 
        next_message_timestamp_str=TIMESTAMP_STR_MINUS_5S,
        conversation_id="conv_missing_content", message_index=1,
        remove_keys=["content"]
    )
    doc_missing_content = Document(page_content=mock_doc_data_missing_content["page_content"], metadata=mock_doc_data_missing_content["metadata"])
    
    # Create a document with all required metadata
    valid_doc = Document(
        page_content=FS_MOCK_AGENT_DOC_1["page_content"],
        metadata=FS_MOCK_AGENT_DOC_1["metadata"].copy()
    )
    
    # Mock the similarity search to return one valid and one invalid document
    mock_agent_store.asimilarity_search = AsyncMock(return_value=[doc_missing_content, valid_doc])

    # The function should raise an error since we asked for 2 documents but only 1 is valid
    with pytest.raises(ValueError, match="Not enough valid documents retrieved from vector store"):
        await RagAgent._get_few_shot_examples(
            conversation_history=mock_history,
            vector_store=mock_agent_store,
            k=2
        )
    
    mock_agent_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=2)

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_agent_wrong_role_in_metadata():
    mock_agent_store = MagicMock(spec=VectorStore)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 2  # Simulate a non-empty store

    mock_history = [Message(sender=ParticipantRole.CUSTOMER, content="Query for wrong role test", timestamp=datetime.now())]
    formatted_history_query = _format_conversation_history(mock_history)

    # Document with 'customer' role in metadata, should be filtered by get_few_shot_examples_for_agent
    mock_doc_data_wrong_role = create_mock_doc_data(
        history_messages=[Message(sender=ParticipantRole.AGENT, content="History for wrong role doc", timestamp=TIMESTAMP_MINUS_10S)],
        next_message_role=ParticipantRole.CUSTOMER.value,  # This is the "wrong" role for agent examples
        next_message_content="This is a customer message content",
        next_message_timestamp_str=TIMESTAMP_STR_MINUS_5S,
        conversation_id="conv_wrong_role", message_index=1
    )
    doc_wrong_role = Document(page_content=mock_doc_data_wrong_role["page_content"], metadata=mock_doc_data_wrong_role["metadata"])
    
    # A valid document
    valid_doc = Document(
        page_content=FS_MOCK_AGENT_DOC_1["page_content"],
        metadata=FS_MOCK_AGENT_DOC_1["metadata"].copy()
    )
    
    # Mock the similarity search to return one valid and one wrong-role document
    mock_agent_store.asimilarity_search = AsyncMock(return_value=[doc_wrong_role, valid_doc])

    # The function should raise an error since we asked for 2 documents but only 1 has the correct role
    with pytest.raises(ValueError, match="Not enough valid documents retrieved from vector store"):
        await RagAgent._get_few_shot_examples(
            conversation_history=mock_history,
            vector_store=mock_agent_store,
            k=2
        )
    
    mock_agent_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=2)

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_agent_similarity_search_error():
    mock_agent_store = MagicMock(spec=VectorStore)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 1
    mock_agent_store.asimilarity_search = AsyncMock(side_effect=Exception("DB error"))

    mock_history = [Message(sender=ParticipantRole.CUSTOMER, content="Query causing error", timestamp=datetime.now())]
    formatted_history_query = _format_conversation_history(mock_history)

    with pytest.raises(Exception, match="DB error"):
        await RagAgent._get_few_shot_examples(
            conversation_history=mock_history,
            vector_store=mock_agent_store,
            k=2
        )
    mock_agent_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=2)
