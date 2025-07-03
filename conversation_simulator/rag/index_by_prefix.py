
import logging
from datetime import datetime, timedelta
from collections.abc import Sequence

from langchain_core.documents import Document

from langchain_core.vectorstores import VectorStore
from ..models import Conversation, Message, ParticipantRole

logger = logging.getLogger(__name__)


def _format_conversation_history(messages: Sequence[Message]) -> str:
    """Formats a list of messages into a single string."""
    return "\n".join([f"{msg.sender.value.capitalize()}: {msg.content}" for msg in messages])


def conversations_to_langchain_documents(
    conversations: list[Conversation]
) -> list[Document]:
    documents: list[Document] = []
    # Filter out empty conversations
    conversations = [conversation for conversation in conversations if len(conversation.messages) > 0]

    for conversation in conversations:
        for i in range(0, len(conversation.messages)):
            current_message: Message = conversation.messages[i]
            next_message: Message | None = conversation.messages[i+1] if i+1 < len(conversation.messages) else None

            history_messages: list[Message] = list(conversation.messages[:i+1])
            # The history becomes the content to be embedded
            page_content = _format_conversation_history(history_messages)

            full_conversation = _format_conversation_history(conversation.messages)

            # The 'next message' becomes the metadata
            metadata = {
                "current_message_index": i,
                "has_next_message": bool(next_message),
                "current_message_role": current_message.sender.value,
                "current_message_timestamp": current_message.timestamp.isoformat(),
                "full_conversation": full_conversation
            }
            
            if next_message:
                metadata["next_message_role"] = next_message.sender.value
                metadata["next_message_content"] = next_message.content
                metadata["next_message_timestamp"] = next_message.timestamp.isoformat()

            documents.append(Document(page_content=page_content, metadata=metadata))
    return documents


async def get_few_shot_examples(
    conversation_history: Sequence[Message],
    vector_store: VectorStore,
    k: int
) -> list[tuple[Document, float]]:
    """Retrieves k relevant documents for a given role of the current last message."""

    current_message_role = conversation_history[-1].sender
    
    query = _format_conversation_history(conversation_history)

    retrieved_docs: list[tuple[Document, float]] = await vector_store.asimilarity_search_with_score(
        query=query, k=k,
        filter={"current_message_role": current_message_role.value}
    )

    # Sort retrieved docs by score
    retrieved_docs.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate documents coming from the same conversation, by comparing the full_conversation metadata
    unique_docs = []
    seen_conversations = set()
    for doc, score in retrieved_docs:
        if doc.metadata.get("full_conversation") not in seen_conversations:
            unique_docs.append((doc, score))
            seen_conversations.add(doc.metadata.get("full_conversation"))

    logger.info(f"Retrieved {len(retrieved_docs)} documents, deduplicated to {len(unique_docs)}.")

    return unique_docs


async def probability_of_next_message_for(role: ParticipantRole, similar_docs: list[tuple[Document, float]]) -> float:
    """Estimates the probability of a next message for a given role based on similar documents.
    
    This function uses a weighted approach where each document's contribution is
    weighted by its similarity score, giving more influence to documents that are
    more similar to the query.
    
    Args:
        role: The participant role to calculate probability for.
        similar_docs: List of (document, similarity_score) tuples.
        
    Returns:
        A float between 0 and 1 representing the probability.
    """
    if not similar_docs:
        return 0.0
    
    total_weight = sum(score for _, score in similar_docs)
    if total_weight == 0:
        return 0.0
    
    # Sum the weights of documents where the next message's role matches our target
    weighted_matches = sum(
        score for doc, score in similar_docs
        if doc.metadata.get("has_next_message", False) and 
           doc.metadata.get("next_message_role") == role.value
    )
    
    return float(weighted_matches / total_weight)


async def calculate_next_message_timedeltas(role: ParticipantRole, similar_docs: list[tuple[Document, float]]) -> list[float]:
    """Calculate the timedelta in seconds between current last message and next message for a given role (of the next message).
    
    Returns a list of timedeltas for each document, for documents where the next message's role matches the target role.
    Order is the same as in the input list.
    
    Args:
        role: The participant role of the next message to calculate time delta for.
        similar_docs: List of (document, similarity_score) tuples.
        
    Returns:
        List of timedeltas in seconds, or empty list if no valid docs are available.
    """
    relevant_docs = [(doc, score) for doc, score in similar_docs if doc.metadata.get("next_message_role") == role.value]
    
    def timedelta_to_next_message(doc: Document) -> float:
        time_delta: timedelta = (
            datetime.fromisoformat(doc.metadata.get("next_message_timestamp")) -
            datetime.fromisoformat(doc.metadata.get("current_message_timestamp"))
        )
        return time_delta.total_seconds()
    
    return [timedelta_to_next_message(doc) for doc, _ in relevant_docs]
    