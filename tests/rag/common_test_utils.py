from datetime import datetime, timedelta
from typing import TypedDict, Any

from conversation_simulator.models import Message, ParticipantRole
from conversation_simulator.models.conversation import Conversation
from conversation_simulator.rag.commons import _format_conversation_history

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
    metadata: dict[str, Any]  # Use Dict for flexibility in tests

def create_mock_doc_data(
    history_messages: list[Message],
    next_message_role: str,
    next_message_content: str,
    next_message_timestamp_str: str,
    conversation_id: str,
    message_index: int,
    remove_keys: list[str] | None = None
) -> MockDocData:
    metadata: dict[str, Any] = {
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
BASE_TEST_TIME = datetime(2023, 1, 1, 12, 0, 0)

MOCK_CONVERSATIONS: list[Conversation] = [
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

# --- Mock Langchain Document objects for Few-Shot (FS) example tests ---
FS_TIMESTAMP_STR = TIMESTAMP_NOW.isoformat()

FS_MOCK_AGENT_DOC_1: MockDocData = {
    "page_content": "Agent response 1 to C1",
    "metadata": {
        "conversation_id": "fs_conv_1",
        "message_index": 1,
        "role": ParticipantRole.AGENT.value,
        "content": "Agent response 1 to C1",
        "timestamp": FS_TIMESTAMP_STR,
    },
}

FS_MOCK_AGENT_DOC_2: MockDocData = {
    "page_content": "Agent response 2 to C1FU",
    "metadata": {
        "conversation_id": "fs_conv_1",
        "message_index": 3,
        "role": ParticipantRole.AGENT.value,
        "content": "Agent response 2 to C1FU",
        "timestamp": FS_TIMESTAMP_STR,
    },
}

FS_MOCK_AGENT_DOC_3: MockDocData = {
    "page_content": "Agent response 3 after clarifications",
    "metadata": {
        "conversation_id": "fs_conv_3",
        "message_index": 3,
        "role": ParticipantRole.AGENT.value,
        "content": "Agent response 3 after clarifications",
        "timestamp": FS_TIMESTAMP_STR
    }
}

FS_MOCK_AGENT_DOC_4_FOLLOWUP: MockDocData = {
    "page_content": "Agent A4 Followup",
    "metadata": {
        "conversation_id": "fs_conv_4",
        "message_index": 3,
        "role": ParticipantRole.AGENT.value,
        "content": "Agent A4 Followup",
        "timestamp": FS_TIMESTAMP_STR
    }
}

FS_MOCK_AGENT_DOC_PROACTIVE: MockDocData = {
    "page_content": "Agent proactive outreach",
    "metadata": {
        "conversation_id": "fs_conv_2",
        "message_index": 0,
        "role": ParticipantRole.AGENT.value,
        "timestamp": FS_TIMESTAMP_STR
    }
}

FS_MOCK_CUSTOMER_DOC_1: MockDocData = {
    "page_content": "Customer query 1",
    "metadata": {
        "conversation_id": "fs_conv_1",
        "message_index": 2,
        "role": ParticipantRole.CUSTOMER.value,
        "content": "Customer follow-up 1",
        "timestamp": FS_TIMESTAMP_STR
    }
}

FS_MOCK_CUSTOMER_DOC_2: MockDocData = {
    "page_content": "Agent response 1 to C1",
    "metadata": {
        "conversation_id": "fs_conv_1",
        "message_index": 0,
        "role": ParticipantRole.CUSTOMER.value,
        "content": "Hello, I have a problem.",
        "timestamp": FS_TIMESTAMP_STR
    }
}

FS_MOCK_CUSTOMER_DOC_WRONG_ROLE: MockDocData = {
    "page_content": "Agent A4 Followup",
    "metadata": {
        "conversation_id": "fs_conv_4",
        "message_index": 3,
        "role": ParticipantRole.AGENT.value,
        "timestamp": FS_TIMESTAMP_STR
    }
}

FS_MOCK_CUSTOMER_DOC_MISSING_TIMESTAMP: MockDocData = {
    "page_content": "Customer query for missing ts",
    "metadata": {
        "conversation_id": "fs_conv_ts",
        "message_index": 0,
        "role": ParticipantRole.CUSTOMER.value,
        "content": "Content for missing ts"
    }
}
