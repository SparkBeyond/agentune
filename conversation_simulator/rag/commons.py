
import logging
from typing import List, Sequence

from langchain_core.documents import Document

from langchain_core.vectorstores import VectorStore
from ..models import Conversation, Message, ParticipantRole

logger = logging.getLogger(__name__)


def _format_conversation_history(messages: Sequence[Message]) -> str:
    """Formats a list of messages into a single string."""
    return "\n".join([f"{msg.sender.value.capitalize()}: {msg.content}" for msg in messages])

# Decouple this too. 
def conversations_to_langchain_documents(
    conversations: List[Conversation],
    role: ParticipantRole,
) -> List[Document]:
    documents: List[Document] = []
    for conversation in conversations:
        if len(conversation.messages) < 2:
            continue # Need at least one message for history and one for the 'next message'

        for i in range(1, len(conversation.messages)):
            next_message: Message = conversation.messages[i]
            
            # Skip if the next message's role doesn't match the target role
            if next_message.sender != role:
                continue
                
            if not next_message.content:
                logger.debug(f"Skipping empty next_message at index {i} in conversation starting with: {conversation.messages[0].content[:50]}...")
                continue

            history_messages: List[Message] = list(conversation.messages[:i])
            # The history becomes the content to be embedded
            page_content = _format_conversation_history(history_messages)

            # The 'next message' becomes the metadata
            metadata = {
                "message_index": i,
                "role": next_message.sender.value,
                "content": next_message.content,  # Store the response content
                "timestamp": next_message.timestamp.isoformat()
            }

            documents.append(Document(page_content=page_content, metadata=metadata))
    return documents

async def get_few_shot_examples(
    conversation_history: Sequence[Message],
    vector_store: VectorStore,
    k: int,
    target_role: ParticipantRole,
) -> List[Document]:
    """Retrieves k relevant documents for a given role from a vector store."""
    query = _format_conversation_history(conversation_history)

    # Let exceptions propagate instead of catching them
    retrieved_docs: List[Document] = await vector_store.asimilarity_search(
        query=query, k=k
    )

    if not retrieved_docs: 
        raise ValueError("No documents retrieved from vector store.")

    # Check that the retrieved documents have the correct metadata using list comprehension

    valid_docs = [
        doc for doc in retrieved_docs
        if all(key in doc.metadata for key in ["role", "content", "timestamp"])
        and doc.metadata.get("role") == target_role.value
    ]

    if len(valid_docs) < k: 
        raise ValueError(f"Not enough valid documents retrieved from vector store. Expected {k}, got {len(valid_docs)}.")
    
    return valid_docs
