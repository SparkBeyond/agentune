import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import TypedDict, Dict, Any, List, Optional
from datetime import datetime, timedelta
from conversation_simulator.models import Message, ParticipantRole
from conversation_simulator.models.conversation import Conversation

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from conversation_simulator.rag.processing import _format_conversation_history


# --- Timestamps ---
TIMESTAMP_NOW = datetime.now()
TIMESTAMP_STR_NOW = TIMESTAMP_NOW.isoformat()
TIMESTAMP_MINUS_5S = TIMESTAMP_NOW - timedelta(seconds=5)
TIMESTAMP_STR_MINUS_5S = TIMESTAMP_MINUS_5S.isoformat()
TIMESTAMP_MINUS_10S = TIMESTAMP_NOW - timedelta(seconds=10)
TIMESTAMP_STR_MINUS_10S = TIMESTAMP_MINUS_10S.isoformat()
TIMESTAMP_MINUS_15S = TIMESTAMP_NOW - timedelta(seconds=15)
TIMESTAMP_STR_MINUS_15S = TIMESTAMP_MINUS_15S.isoformat()


# --- Mock Document Data Structure ---
class MockDocMetadata(TypedDict):
    conversation_id: str
    message_index: int
    role: str
    content: str
    timestamp: str

class MockDocData(TypedDict):
    page_content: str
    metadata: Dict[str, Any] # Use Dict for flexibility in tests

# --- Helper function to create mock document data ---
def create_mock_doc_data(
    history_messages: List[Message],
    next_message_role: str, # Use str directly as it's stored as string in metadata
    next_message_content: str,
    next_message_timestamp_str: str,
    conversation_id: str,
    message_index: int,
    remove_keys: Optional[List[str]] = None
) -> MockDocData:
    metadata: Dict[str, Any] = { # Use Dict[str, Any] for flexibility before TypedDict conversion
        "conversation_id": conversation_id,
        "message_index": message_index,
        "role": next_message_role,
        "content": next_message_content,
        "timestamp": next_message_timestamp_str,
    }
    if remove_keys:
        for key_to_remove in remove_keys:
            if key_to_remove in metadata:
                del metadata[key_to_remove]

    return {
        "page_content": _format_conversation_history(history_messages),
        "metadata": metadata
    }

# --- Mock Agent Documents ---
MOCK_AGENT_DOCS = [
    create_mock_doc_data(
        history_messages=[Message(sender=ParticipantRole.CUSTOMER, content="Hello Agent MOCK_AGENT_DOC_1", timestamp=TIMESTAMP_MINUS_10S)],
        next_message_role=ParticipantRole.AGENT.value,
        next_message_content="Agent response 1 to C1",
        next_message_timestamp_str=TIMESTAMP_STR_MINUS_5S,
        conversation_id="conv_agent_1", message_index=1
    ),
    create_mock_doc_data(
        history_messages=[Message(sender=ParticipantRole.CUSTOMER, content="Customer query for MOCK_AGENT_DOC_2", timestamp=TIMESTAMP_MINUS_10S)],
        next_message_role=ParticipantRole.AGENT.value,
        next_message_content="Agent response regarding X",
        next_message_timestamp_str=TIMESTAMP_STR_MINUS_5S,
        conversation_id="conv_agent_2", message_index=1
    ),
    create_mock_doc_data(
        history_messages=[
            Message(sender=ParticipantRole.CUSTOMER, content="Initial question for multi-turn", timestamp=TIMESTAMP_MINUS_15S),
            Message(sender=ParticipantRole.AGENT, content="First agent reply in multi-turn", timestamp=TIMESTAMP_MINUS_10S),
            Message(sender=ParticipantRole.CUSTOMER, content="Customer follow-up for MOCK_AGENT_DOC_3", timestamp=TIMESTAMP_MINUS_5S),
        ],
        next_message_role=ParticipantRole.AGENT.value,
        next_message_content="Agent's detailed answer after follow-up",
        next_message_timestamp_str=TIMESTAMP_STR_NOW,
        conversation_id="conv_agent_3_multi", message_index=3
    ),
    create_mock_doc_data(
        history_messages=[
            Message(sender=ParticipantRole.CUSTOMER, content="Tell me about product X", timestamp=TIMESTAMP_MINUS_10S), 
            Message(sender=ParticipantRole.AGENT, content="Agent response 1 to C1", timestamp=TIMESTAMP_MINUS_5S), 
            Message(sender=ParticipantRole.CUSTOMER, content="Thanks, what about Y?", timestamp=TIMESTAMP_NOW) 
        ],
        next_message_role=ParticipantRole.AGENT.value,
        next_message_content="Follow-up agent response about Y", 
        next_message_timestamp_str=(TIMESTAMP_NOW + timedelta(seconds=5)).isoformat(),
        conversation_id="conv_agent_1", message_index=3 
    ),
]

# --- Mock Customer Documents ---
MOCK_CUSTOMER_DOCS = [
    create_mock_doc_data(
        history_messages=[Message(sender=ParticipantRole.AGENT, content="Hello Customer MOCK_CUSTOMER_DOC_1", timestamp=TIMESTAMP_MINUS_10S)],
        next_message_role=ParticipantRole.CUSTOMER.value,
        next_message_content="Customer response 1 to A1",
        next_message_timestamp_str=TIMESTAMP_STR_MINUS_5S,
        conversation_id="conv_cust_1", message_index=1
    ),
    create_mock_doc_data(
        history_messages=[Message(sender=ParticipantRole.AGENT, content="Agent query for MOCK_CUSTOMER_DOC_2", timestamp=TIMESTAMP_MINUS_10S)],
        next_message_role=ParticipantRole.CUSTOMER.value,
        next_message_content="Customer response about Z",
        next_message_timestamp_str=TIMESTAMP_STR_MINUS_5S,
        conversation_id="conv_cust_2", message_index=1
    ),
]

# Mock conversation data for create_vector_stores tests
BASE_TEST_TIME = datetime(2023, 1, 1, 12, 0, 0)  # Define a fixed base time for timestamps

MOCK_CONVERSATIONS: List[Conversation] = [
    Conversation(
        messages=tuple([
            Message(sender=ParticipantRole.CUSTOMER, content="Hello, I have a problem.", timestamp=BASE_TEST_TIME),
            Message(sender=ParticipantRole.AGENT, content="Hi, how can I help you?", timestamp=BASE_TEST_TIME + timedelta(seconds=5)),
            Message(sender=ParticipantRole.CUSTOMER, content="My order hasn't arrived.", timestamp=BASE_TEST_TIME + timedelta(seconds=10)),
        ]),
    ),
    Conversation(
        messages=tuple([
            Message(sender=ParticipantRole.CUSTOMER, content="I want to return an item.", timestamp=BASE_TEST_TIME + timedelta(seconds=15)),
            Message(sender=ParticipantRole.AGENT, content="Sure, what is the order number?", timestamp=BASE_TEST_TIME + timedelta(seconds=20)),
        ]),
    ),
]


# --- Unit Tests for get_few_shot_examples_for_agent ---

# Mock Langchain Document objects that would be returned by similarity search

TIMESTAMP_STR = datetime.now().isoformat()

FS_MOCK_AGENT_DOC_1: MockDocData = {
    "page_content": "Agent response 1 to C1",
    "metadata": {
        "conversation_id": "fs_conv_1", 
        "message_index": 1, # Example index
        "role": ParticipantRole.AGENT.value,
        "content": "Agent response 1 to C1", # Added content
        "timestamp": TIMESTAMP_STR,
    },
}
FS_MOCK_AGENT_DOC_2: MockDocData = {
    "page_content": "Agent response 2 to C1FU",
    "metadata": {
        "conversation_id": "fs_conv_1", 
        "message_index": 3, # Example index
        "role": ParticipantRole.AGENT.value,
        "content": "Agent response 2 to C1FU", # Added content
        "timestamp": TIMESTAMP_STR,
    },
}
FS_MOCK_AGENT_DOC_3: MockDocData = {
    "page_content": "Agent response 3 after clarifications",
    "metadata": {
        "conversation_id": "fs_conv_3",
        "message_index": 3,
        "role": ParticipantRole.AGENT.value,
        "content": "Agent response 3 after clarifications",
        "timestamp": TIMESTAMP_STR
    }
}
FS_MOCK_AGENT_DOC_4_FOLLOWUP: MockDocData = {
    "page_content": "Agent A4 Followup",
    "metadata": {
        "conversation_id": "fs_conv_4",
        "message_index": 3,
        "role": ParticipantRole.AGENT.value,
        "content": "Agent A4 Followup",
        "timestamp": TIMESTAMP_STR
    }
}
FS_MOCK_AGENT_DOC_PROACTIVE: MockDocData = {
    "page_content": "Agent proactive outreach",
    "metadata": {"conversation_id": "fs_conv_2", "message_index": 0, "role": "agent", "timestamp": TIMESTAMP_STR}
}

# Mock Langchain Document objects for CUSTOMER messages
FS_MOCK_CUSTOMER_DOC_1: MockDocData = {
    "page_content": "Customer query 1", # This is the history *before* this customer message
    "metadata": {"conversation_id": "fs_conv_1", "message_index": 2, "role": "customer", "content": "Customer follow-up 1", "timestamp": TIMESTAMP_STR}
}
FS_MOCK_CUSTOMER_DOC_2: MockDocData = {
    "page_content": "Agent response 1 to C1", # History before this customer message
    "metadata": {"conversation_id": "fs_conv_1", "message_index": 0, "role": "customer", "content": "Hello, I have a problem.", "timestamp": TIMESTAMP_STR}
}
# For testing filtering
FS_MOCK_CUSTOMER_DOC_WRONG_ROLE: MockDocData = { # This doc has AGENT role, should be filtered out by get_few_shot_examples_for_customer
    "page_content": "Agent A4 Followup",
    "metadata": {"conversation_id": "fs_conv_4", "message_index": 3, "role": "agent", "timestamp": TIMESTAMP_STR}
}
FS_MOCK_CUSTOMER_DOC_MISSING_TIMESTAMP: MockDocData = {
    "page_content": "Customer query for missing ts",
    "metadata": {"conversation_id": "fs_conv_ts", "message_index": 0, "role": "customer", "content": "Content for missing ts"} # No timestamp
}


@pytest.mark.asyncio
async def test_get_few_shot_examples_success():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_agent, _format_conversation_history # Import locally
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

    examples = await get_few_shot_examples_for_agent(
        conversation_history=mock_history,
        agent_vector_store=mock_agent_store,
        k=2
    )

    assert len(examples) == 2
    assert isinstance(examples[0], Document)
    assert examples[0].metadata["role"] == ParticipantRole.AGENT.value
    assert examples[0].metadata["content"] == FS_MOCK_AGENT_DOC_1["metadata"]["content"]
    assert examples[1].metadata["content"] == FS_MOCK_AGENT_DOC_4_FOLLOWUP["metadata"]["content"]
    
    mock_agent_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=2)
@pytest.mark.asyncio
async def test_get_few_shot_examples_less_than_k_valid():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_agent, _format_conversation_history
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

    examples = await get_few_shot_examples_for_agent(
        conversation_history=mock_history,
        agent_vector_store=mock_agent_store,
        k=3
    )

    assert len(examples) == 1
    assert isinstance(examples[0], Document)
    assert examples[0].metadata["content"] == FS_MOCK_AGENT_DOC_1["metadata"]["content"]
    mock_agent_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=3)

@pytest.mark.asyncio
async def test_get_few_shot_examples_no_valid_pairs_formed():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_agent, _format_conversation_history
    mock_agent_store = MagicMock(spec=VectorStore)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 2

    mock_history = [Message(sender=ParticipantRole.CUSTOMER, content="Query that finds docs without timestamps", timestamp=datetime.now())]
    formatted_history_query = _format_conversation_history(mock_history)

    # Retrieved docs will all be missing timestamps in metadata, so they will be filtered out.
    mock_doc_data_A_no_ts = create_mock_doc_data(
        history_messages=[Message(sender=ParticipantRole.CUSTOMER, content="History for doc A no ts", timestamp=TIMESTAMP_MINUS_10S)],
        next_message_role=ParticipantRole.AGENT.value, next_message_content="Agent content A no ts",
        next_message_timestamp_str=TIMESTAMP_STR_MINUS_5S, conversation_id="conv_A_no_ts", message_index=1,
        remove_keys=["timestamp"]
    )
    mock_doc_data_B_no_ts = create_mock_doc_data(
        history_messages=[Message(sender=ParticipantRole.CUSTOMER, content="History for doc B no ts", timestamp=TIMESTAMP_MINUS_10S)],
        next_message_role=ParticipantRole.AGENT.value, next_message_content="Agent content B no ts",
        next_message_timestamp_str=TIMESTAMP_STR_MINUS_5S, conversation_id="conv_B_no_ts", message_index=1,
        remove_keys=["timestamp"]
    )
    retrieved_docs = [
        Document(page_content=mock_doc_data_A_no_ts["page_content"], metadata=mock_doc_data_A_no_ts["metadata"]),
        Document(page_content=mock_doc_data_B_no_ts["page_content"], metadata=mock_doc_data_B_no_ts["metadata"])
    ]
    mock_agent_store.asimilarity_search = AsyncMock(return_value=retrieved_docs)

    examples = await get_few_shot_examples_for_agent(
        conversation_history=mock_history,
        agent_vector_store=mock_agent_store,
        k=2
    )
    assert len(examples) == 0
    mock_agent_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=2)

@pytest.mark.asyncio
async def test_get_few_shot_examples_empty_store():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_agent
    mock_agent_store_empty = MagicMock(spec=VectorStore)
    mock_agent_store_empty.index = MagicMock()
    mock_agent_store_empty.index.ntotal = 0 # Empty store

    mock_history = [Message(sender=ParticipantRole.CUSTOMER, content="Any query", timestamp=datetime.now())]
    # from conversation_simulator.rag.processing import _format_conversation_history # Already imported or will be ensured

    examples = await get_few_shot_examples_for_agent(
        conversation_history=mock_history,
        agent_vector_store=mock_agent_store_empty,
        k=2
    )
    assert len(examples) == 0

    examples_none_store = await get_few_shot_examples_for_agent(
        conversation_history=mock_history,
        agent_vector_store=None, # Store is None
        k=2
    )
    assert len(examples_none_store) == 0

@pytest.mark.asyncio
async def test_get_few_shot_examples_empty_conversations_data():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_agent
    mock_agent_store = MagicMock(spec=VectorStore)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 1
    mock_agent_store.asimilarity_search = AsyncMock(return_value=[]) # Search can return empty

    mock_history = [Message(sender=ParticipantRole.CUSTOMER, content="A query", timestamp=datetime.now())]
    from conversation_simulator.rag.processing import _format_conversation_history # Ensure import for this scope if not global
    formatted_history_query = _format_conversation_history(mock_history)

    examples = await get_few_shot_examples_for_agent(
        conversation_history=mock_history,
        agent_vector_store=mock_agent_store,
        k=2
    )
    assert len(examples) == 0
    mock_agent_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=2)

@pytest.mark.asyncio
async def test_get_few_shot_examples_similarity_search_error():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_agent
    mock_agent_store = MagicMock(spec=VectorStore)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 1
    mock_agent_store.asimilarity_search = AsyncMock(side_effect=Exception("DB error"))

    mock_history = [Message(sender=ParticipantRole.CUSTOMER, content="Query causing error", timestamp=datetime.now())]
    from conversation_simulator.rag.processing import _format_conversation_history # Ensure import for this scope if not global
    formatted_history_query = _format_conversation_history(mock_history)

    examples = await get_few_shot_examples_for_agent(
        conversation_history=mock_history,
        agent_vector_store=mock_agent_store,
        k=2
    )
    assert len(examples) == 0
    mock_agent_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=2)

@pytest.mark.asyncio
async def test_get_few_shot_examples_missing_metadata():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_agent, _format_conversation_history
    mock_agent_store = MagicMock(spec=VectorStore)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 2 # Simulate a non-empty store

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

    # A valid document
    valid_doc = Document(page_content=FS_MOCK_AGENT_DOC_1["page_content"], metadata=FS_MOCK_AGENT_DOC_1["metadata"])

    mock_agent_store.asimilarity_search = AsyncMock(return_value=[doc_missing_content, valid_doc])

    examples = await get_few_shot_examples_for_agent(
        conversation_history=mock_history,
        agent_vector_store=mock_agent_store,
        k=3
    )
    # Only the valid_doc should produce an example
    assert len(examples) == 1
    assert isinstance(examples[0], Document)
    assert examples[0].page_content == FS_MOCK_AGENT_DOC_1["page_content"]
    assert examples[0].metadata["content"] == FS_MOCK_AGENT_DOC_1["metadata"]["content"]
    assert examples[0].metadata["role"] == ParticipantRole.AGENT.value
    mock_agent_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=3)

@pytest.mark.asyncio
async def test_get_few_shot_examples_wrong_role_in_metadata(): # Renamed test
    from conversation_simulator.rag.processing import get_few_shot_examples_for_agent, _format_conversation_history
    mock_agent_store = MagicMock(spec=VectorStore)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 2 # Simulate a non-empty store

    mock_history = [Message(sender=ParticipantRole.CUSTOMER, content="Query for wrong role test", timestamp=datetime.now())]
    formatted_history_query = _format_conversation_history(mock_history)

    # Document with 'customer' role in metadata, should be filtered by get_few_shot_examples_for_agent
    mock_doc_data_wrong_role = create_mock_doc_data(
        history_messages=[Message(sender=ParticipantRole.AGENT, content="History for wrong role doc", timestamp=TIMESTAMP_MINUS_10S)],
        next_message_role=ParticipantRole.CUSTOMER.value, # This is the "wrong" role
        next_message_content="This is a customer message content",
        next_message_timestamp_str=TIMESTAMP_STR_MINUS_5S,
        conversation_id="conv_wrong_role", message_index=1
    )
    doc_wrong_role = Document(page_content=mock_doc_data_wrong_role["page_content"], metadata=mock_doc_data_wrong_role["metadata"])

    # A valid document (FS_MOCK_AGENT_DOC_1 is already correctly structured)
    valid_doc = Document(
        page_content=FS_MOCK_AGENT_DOC_1["page_content"],
        metadata=FS_MOCK_AGENT_DOC_1["metadata"]
    )

    mock_agent_store.asimilarity_search = AsyncMock(return_value=[doc_wrong_role, valid_doc])

    examples = await get_few_shot_examples_for_agent(
        conversation_history=mock_history,
        agent_vector_store=mock_agent_store,
        k=2
    )
    # Only the valid_doc should produce an example
    assert len(examples) == 1
    assert isinstance(examples[0], Document)
    assert examples[0].page_content == FS_MOCK_AGENT_DOC_1["page_content"]
    assert examples[0].metadata["content"] == FS_MOCK_AGENT_DOC_1["metadata"]["content"]
    assert examples[0].metadata["role"] == ParticipantRole.AGENT.value
    mock_agent_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=2)


# --- Unit Tests for get_few_shot_examples_for_customer ---

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_customer_success():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_customer, _format_conversation_history # Import locally
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

    examples = await get_few_shot_examples_for_customer(
        conversation_history=mock_history,
        customer_vector_store=mock_customer_store,
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
    from conversation_simulator.rag.processing import get_few_shot_examples_for_customer
    mock_customer_store_empty = MagicMock(spec=FAISS)
    mock_customer_store_empty.index = MagicMock()
    mock_customer_store_empty.index.ntotal = 0 # Empty store

    mock_history = [Message(sender=ParticipantRole.AGENT, content="Any query for empty store", timestamp=datetime.now())]

    examples = await get_few_shot_examples_for_customer(
        conversation_history=mock_history,
        customer_vector_store=mock_customer_store_empty,
        k=2
    )
    assert len(examples) == 0

    examples_none_store = await get_few_shot_examples_for_customer(
        conversation_history=mock_history,
        customer_vector_store=None, # Store is None
        k=2
    )
    assert len(examples_none_store) == 0

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_customer_no_docs_retrieved():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_customer, _format_conversation_history
    mock_customer_store = MagicMock(spec=VectorStore)
    mock_customer_store.index = MagicMock()
    mock_customer_store.index.ntotal = 1 # Simulate non-empty store
    mock_customer_store.asimilarity_search = AsyncMock(return_value=[]) # Search returns empty list

    mock_history = [Message(sender=ParticipantRole.AGENT, content="A query, no docs", timestamp=datetime.now())]
    formatted_history_query = _format_conversation_history(mock_history)

    examples = await get_few_shot_examples_for_customer(
        conversation_history=mock_history,
        customer_vector_store=mock_customer_store,
        k=2
    )
    assert len(examples) == 0
    mock_customer_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=2)

@pytest.mark.asyncio
async def test_get_few_shot_examples_for_customer_similarity_search_error():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_customer, _format_conversation_history
    mock_customer_store = MagicMock(spec=VectorStore)
    mock_customer_store.index = MagicMock()
    mock_customer_store.index.ntotal = 1 # Simulate non-empty store
    mock_customer_store.asimilarity_search = AsyncMock(side_effect=Exception("Customer DB error"))

    mock_history = [Message(sender=ParticipantRole.AGENT, content="Query causing customer DB error", timestamp=datetime.now())]
    formatted_history_query = _format_conversation_history(mock_history)

    examples = await get_few_shot_examples_for_customer(
        conversation_history=mock_history,
        customer_vector_store=mock_customer_store,
        k=2
    )
    assert len(examples) == 0
    mock_customer_store.asimilarity_search.assert_called_once_with(query=formatted_history_query, k=2)
