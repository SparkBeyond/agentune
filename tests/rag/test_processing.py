import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import TypedDict, Dict, Any

from langchain_core.documents import Document
from pydantic import SecretStr
from langchain_community.vectorstores import FAISS
from conversation_simulator.rag import (
    create_vector_stores_from_conversations,
)

# Mock conversation data
MOCK_CONVERSATIONS = [
    {
        "id": "conv_1",
        "messages": [
            {"role": "customer", "content": "Hello, I have a problem."},
            {"role": "agent", "content": "Hi, how can I help you?"},
            {"role": "customer", "content": "My order hasn't arrived."},
        ],
    },
    {
        "id": "conv_2",
        "messages": [
            {"role": "customer", "content": "I want to return an item."},
            {"role": "agent", "content": "Sure, what is the order number?"},
        ],
    },
]

@pytest.mark.asyncio
@patch("conversation_simulator.rag.processing.FAISS.afrom_documents")
@patch("conversation_simulator.rag.processing.OpenAIEmbeddings")
async def test_create_vector_stores_from_conversations_success(
    mock_openai_embeddings: MagicMock,
    mock_faiss_afrom_documents: MagicMock,
):
    """
    Tests successful creation of vector stores from conversations using mocked OpenAI.
    """
    # Arrange
    # Mock FAISS.afrom_documents to return a mock FAISS instance
    mock_faiss_instance = MagicMock(spec=FAISS) # Use spec for better mocking
    mock_faiss_afrom_documents.return_value = mock_faiss_instance
    
    # Mock the OpenAIEmbeddings class instance
    mock_embedding_instance = MagicMock()
    mock_openai_embeddings.return_value = mock_embedding_instance

    # Act
    customer_store, agent_store = await create_vector_stores_from_conversations(
        conversations=MOCK_CONVERSATIONS,
        openai_api_key="fake-api-key",
    )

    # Assert
    # Check that the function returns two FAISS instances (our mocked one)
    assert customer_store == mock_faiss_instance
    assert agent_store == mock_faiss_instance
    # Optionally, also check type if the mock spec isn't strict enough for your liking
    assert isinstance(customer_store, FAISS)
    assert isinstance(agent_store, FAISS)

    # Check that OpenAIEmbeddings was instantiated correctly
    mock_openai_embeddings.assert_called_once_with(
        api_key=SecretStr("fake-api-key"), model="text-embedding-ada-002"
    )

    # Check that FAISS.afrom_documents was called twice (once for customer, once for agent)
    assert mock_faiss_afrom_documents.call_count == 2
    
    # Check the call for customer documents
    customer_call_args = mock_faiss_afrom_documents.call_args_list[0].kwargs
    assert len(customer_call_args['documents']) == 3  # 3 customer messages
    assert customer_call_args['embedding'] == mock_embedding_instance
    assert customer_call_args['documents'][0].page_content == "Hello, I have a problem."
    assert customer_call_args['documents'][0].metadata["role"] == "customer"

    # Check the call for agent documents
    agent_call_args = mock_faiss_afrom_documents.call_args_list[1].kwargs
    assert len(agent_call_args['documents']) == 2  # 2 agent messages
    assert agent_call_args['embedding'] == mock_embedding_instance
    assert agent_call_args['documents'][0].page_content == "Hi, how can I help you?"
    assert agent_call_args['documents'][0].metadata["role"] == "agent"


@pytest.mark.asyncio
async def test_create_vector_stores_no_conversations():
    """
    Tests that a ValueError is raised if no conversations are provided.
    """
    with pytest.raises(ValueError, match="No conversations provided"):
        await create_vector_stores_from_conversations(
            conversations=[],
            openai_api_key="fake-api-key",
        )


@pytest.mark.asyncio
async def test_create_vector_stores_no_api_key():
    """
    Tests that a ValueError is raised if no OpenAI API key is provided.
    """
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        await create_vector_stores_from_conversations(
            conversations=MOCK_CONVERSATIONS,
            openai_api_key="",
        )


# --- Unit Tests for get_few_shot_examples_for_agent ---

# More detailed mock conversations for testing few-shot example retrieval
MOCK_CONVERSATIONS_FOR_FEW_SHOT = [
    {
        "id": "fs_conv_1", # Standard case
        "messages": [
            {"role": "customer", "content": "Customer query 1"},
            {"role": "agent", "content": "Agent response 1 to C1"}, # Target for retrieval
            {"role": "customer", "content": "Customer follow-up 1"},
            {"role": "agent", "content": "Agent response 2 to C1FU"}, # Target for retrieval
        ],
    },
    {
        "id": "fs_conv_2", # Agent message is first
        "messages": [
            {"role": "agent", "content": "Agent proactive outreach"}, # Should not form a pair
            {"role": "customer", "content": "Customer reply to outreach"},
        ],
    },
    {
        "id": "fs_conv_3", # Preceding message is also agent
        "messages": [
            {"role": "customer", "content": "Customer initial query 3"},
            {"role": "agent", "content": "Agent clarification 1"},
            {"role": "agent", "content": "Agent clarification 2 (precedes target)"}, # Precedes target, but is agent
            {"role": "agent", "content": "Agent response 3 after clarifications"}, # Target for retrieval
        ],
    },
    {
        "id": "fs_conv_4", # Valid pair, but later in conversation
        "messages": [
            {"role": "customer", "content": "Customer Q4"},
            {"role": "agent", "content": "Agent A4"},
            {"role": "customer", "content": "Customer Q4 Followup"}, # Precedes target
            {"role": "agent", "content": "Agent A4 Followup"},      # Target for retrieval
        ]
    },
    {
        "id": "fs_conv_5_empty_preceding", # Preceding customer message is empty
        "messages": [
            {"role": "customer", "content": ""}, # Empty preceding message
            {"role": "agent", "content": "Agent response to empty"}, # Target for retrieval
        ]
    }
]

# Mock Langchain Document objects that would be returned by similarity search
# These correspond to some of the agent messages in MOCK_CONVERSATIONS_FOR_FEW_SHOT

class MockDocData(TypedDict):
    page_content: str
    metadata: Dict[str, Any]

MOCK_AGENT_DOC_1: MockDocData = {
    "page_content": "Agent response 1 to C1",
    "metadata": {"conversation_id": "fs_conv_1", "message_index": 1, "role": "agent"}
}
MOCK_AGENT_DOC_2: MockDocData = {
    "page_content": "Agent response 2 to C1FU",
    "metadata": {"conversation_id": "fs_conv_1", "message_index": 3, "role": "agent"}
}
MOCK_AGENT_DOC_3: MockDocData = {
    "page_content": "Agent response 3 after clarifications",
    "metadata": {"conversation_id": "fs_conv_3", "message_index": 3, "role": "agent"}
}
MOCK_AGENT_DOC_4_FOLLOWUP: MockDocData = {
    "page_content": "Agent A4 Followup",
    "metadata": {"conversation_id": "fs_conv_4", "message_index": 3, "role": "agent"}
}
MOCK_AGENT_DOC_PROACTIVE: MockDocData = {
    "page_content": "Agent proactive outreach",
    "metadata": {"conversation_id": "fs_conv_2", "message_index": 0, "role": "agent"}
}
MOCK_AGENT_DOC_TO_EMPTY: MockDocData = {
    "page_content": "Agent response to empty",
    "metadata": {"conversation_id": "fs_conv_5_empty_preceding", "message_index": 1, "role": "agent"}
}

@pytest.mark.asyncio
async def test_get_few_shot_examples_success():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_agent # Import locally
    mock_agent_store = MagicMock(spec=FAISS)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 3 # Simulate a non-empty store

    # Simulate asimilarity_search returning relevant agent documents
    retrieved_docs = [
        Document(page_content=MOCK_AGENT_DOC_1["page_content"], metadata=MOCK_AGENT_DOC_1["metadata"]),
        Document(page_content=MOCK_AGENT_DOC_4_FOLLOWUP["page_content"], metadata=MOCK_AGENT_DOC_4_FOLLOWUP["metadata"]),
        Document(page_content=MOCK_AGENT_DOC_2["page_content"], metadata=MOCK_AGENT_DOC_2["metadata"]),
    ]
    mock_agent_store.asimilarity_search = AsyncMock(return_value=retrieved_docs)

    examples = await get_few_shot_examples_for_agent(
        last_customer_message_content="Tell me about product X",
        agent_vector_store=mock_agent_store,
        all_conversations=MOCK_CONVERSATIONS_FOR_FEW_SHOT,
        k=2
    )

    assert len(examples) == 2
    # Example 1 (from MOCK_AGENT_DOC_1)
    assert examples[0][0] == {"role": "user", "content": "Customer query 1"}
    assert examples[0][1] == {"role": "assistant", "content": "Agent response 1 to C1"}
    # Example 2 (from MOCK_AGENT_DOC_4_FOLLOWUP)
    assert examples[1][0] == {"role": "user", "content": "Customer Q4 Followup"}
    assert examples[1][1] == {"role": "assistant", "content": "Agent A4 Followup"}
    
    mock_agent_store.asimilarity_search.assert_called_once_with(query="Tell me about product X", k=2*2)

@pytest.mark.asyncio
async def test_get_few_shot_examples_less_than_k_valid():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_agent
    mock_agent_store = MagicMock(spec=FAISS)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 1

    # Only one retrieved doc leads to a valid pair
    retrieved_docs = [
        Document(page_content=MOCK_AGENT_DOC_1["page_content"], metadata=MOCK_AGENT_DOC_1["metadata"]),
        Document(page_content=MOCK_AGENT_DOC_PROACTIVE["page_content"], metadata=MOCK_AGENT_DOC_PROACTIVE["metadata"])
    ]
    mock_agent_store.asimilarity_search = AsyncMock(return_value=retrieved_docs)

    examples = await get_few_shot_examples_for_agent(
        last_customer_message_content="Another query",
        agent_vector_store=mock_agent_store,
        all_conversations=MOCK_CONVERSATIONS_FOR_FEW_SHOT,
        k=3
    )

    assert len(examples) == 1
    assert examples[0][0] == {"role": "user", "content": "Customer query 1"}
    assert examples[0][1] == {"role": "assistant", "content": "Agent response 1 to C1"}

@pytest.mark.asyncio
async def test_get_few_shot_examples_no_valid_pairs_formed():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_agent
    mock_agent_store = MagicMock(spec=FAISS)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 2

    # Retrieved docs don't lead to valid pairs (agent proactive, agent preceded by agent)
    retrieved_docs = [
        Document(page_content=MOCK_AGENT_DOC_PROACTIVE["page_content"], metadata=MOCK_AGENT_DOC_PROACTIVE["metadata"]),
        Document(page_content=MOCK_AGENT_DOC_3["page_content"], metadata=MOCK_AGENT_DOC_3["metadata"])
    ]
    mock_agent_store.asimilarity_search = AsyncMock(return_value=retrieved_docs)

    examples = await get_few_shot_examples_for_agent(
        last_customer_message_content="Query leading to no valid pairs",
        agent_vector_store=mock_agent_store,
        all_conversations=MOCK_CONVERSATIONS_FOR_FEW_SHOT,
        k=2
    )
    assert len(examples) == 0

@pytest.mark.asyncio
async def test_get_few_shot_examples_empty_store():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_agent
    mock_agent_store_empty = MagicMock(spec=FAISS)
    mock_agent_store_empty.index = MagicMock()
    mock_agent_store_empty.index.ntotal = 0 # Empty store

    examples = await get_few_shot_examples_for_agent(
        last_customer_message_content="Any query",
        agent_vector_store=mock_agent_store_empty,
        all_conversations=MOCK_CONVERSATIONS_FOR_FEW_SHOT,
        k=2
    )
    assert len(examples) == 0

    examples_none_store = await get_few_shot_examples_for_agent(
        last_customer_message_content="Any query",
        agent_vector_store=None, # Store is None
        all_conversations=MOCK_CONVERSATIONS_FOR_FEW_SHOT,
        k=2
    )
    assert len(examples_none_store) == 0

@pytest.mark.asyncio
async def test_get_few_shot_examples_empty_conversations_data():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_agent
    mock_agent_store = MagicMock(spec=FAISS)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 1
    mock_agent_store.asimilarity_search = AsyncMock(return_value=[]) # Search can return empty

    examples = await get_few_shot_examples_for_agent(
        last_customer_message_content="A query",
        agent_vector_store=mock_agent_store,
        all_conversations=[], # Empty full conversation data
        k=2
    )
    assert len(examples) == 0

@pytest.mark.asyncio
async def test_get_few_shot_examples_similarity_search_error():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_agent
    mock_agent_store = MagicMock(spec=FAISS)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 1
    mock_agent_store.asimilarity_search = AsyncMock(side_effect=Exception("DB error"))

    examples = await get_few_shot_examples_for_agent(
        last_customer_message_content="Query causing error",
        agent_vector_store=mock_agent_store,
        all_conversations=MOCK_CONVERSATIONS_FOR_FEW_SHOT,
        k=2
    )
    assert len(examples) == 0

@pytest.mark.asyncio
async def test_get_few_shot_examples_preceding_message_empty_content():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_agent
    mock_agent_store = MagicMock(spec=FAISS)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 1

    retrieved_docs = [
        Document(page_content=MOCK_AGENT_DOC_TO_EMPTY["page_content"], metadata=MOCK_AGENT_DOC_TO_EMPTY["metadata"])
    ]
    mock_agent_store.asimilarity_search = AsyncMock(return_value=retrieved_docs)

    examples = await get_few_shot_examples_for_agent(
        last_customer_message_content="Query for agent response to empty",
        agent_vector_store=mock_agent_store,
        all_conversations=MOCK_CONVERSATIONS_FOR_FEW_SHOT,
        k=1
    )
    assert len(examples) == 0 # Preceding customer message content is empty, so pair is skipped


@pytest.mark.asyncio
async def test_get_few_shot_examples_missing_metadata():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_agent
    mock_agent_store = MagicMock(spec=FAISS)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 2 # Simulate a non-empty store with a couple of docs

    # Document missing 'conversation_id'
    doc_missing_conv_id = Document(
        page_content="Agent text for doc missing conv_id", 
        metadata={"message_index": 0, "role": "agent"} # No conversation_id
    )
    # Document missing 'message_index'
    doc_missing_msg_idx = Document(
        page_content="Agent text for doc missing msg_idx", 
        metadata={"conversation_id": "fs_conv_1", "role": "agent"} # No message_index
    )
    # A valid document to ensure processing continues if some are bad
    valid_doc = Document(
        page_content=MOCK_AGENT_DOC_1["page_content"], 
        metadata=MOCK_AGENT_DOC_1["metadata"]
    )

    mock_agent_store.asimilarity_search = AsyncMock(return_value=[doc_missing_conv_id, doc_missing_msg_idx, valid_doc])

    examples = await get_few_shot_examples_for_agent(
        last_customer_message_content="Query for missing metadata",
        agent_vector_store=mock_agent_store,
        all_conversations=MOCK_CONVERSATIONS_FOR_FEW_SHOT,
        k=3 # Request more than the one valid example we expect
    )
    # Only the valid_doc should produce an example
    assert len(examples) == 1
    assert examples[0][0] == {"role": "user", "content": "Customer query 1"}
    assert examples[0][1] == {"role": "assistant", "content": "Agent response 1 to C1"}

@pytest.mark.asyncio
async def test_get_few_shot_examples_conversation_not_found():
    from conversation_simulator.rag.processing import get_few_shot_examples_for_agent
    mock_agent_store = MagicMock(spec=FAISS)
    mock_agent_store.index = MagicMock()
    mock_agent_store.index.ntotal = 2 # Simulate a non-empty store

    doc_unknown_conv_id = Document(
        page_content="Agent text for unknown convo",
        metadata={"conversation_id": "unknown_conv_id", "message_index": 0, "role": "agent"}
    )
    # Add a valid doc to ensure the function continues processing if one doc is bad
    valid_doc = Document(
        page_content=MOCK_AGENT_DOC_1["page_content"], 
        metadata=MOCK_AGENT_DOC_1["metadata"]
    )

    mock_agent_store.asimilarity_search = AsyncMock(return_value=[doc_unknown_conv_id, valid_doc])

    examples = await get_few_shot_examples_for_agent(
        last_customer_message_content="Query for unknown conversation",
        agent_vector_store=mock_agent_store,
        all_conversations=MOCK_CONVERSATIONS_FOR_FEW_SHOT, # This list doesn't contain "unknown_conv_id"
        k=2
    )
    # Only the valid_doc should produce an example
    assert len(examples) == 1
    assert examples[0][0] == {"role": "user", "content": "Customer query 1"}
    assert examples[0][1] == {"role": "assistant", "content": "Agent response 1 to C1"}



