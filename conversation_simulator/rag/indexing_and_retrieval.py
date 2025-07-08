
import logging
from collections.abc import Sequence

from langchain_core.documents import Document

from langchain_core.vectorstores import VectorStore
from ..models import Conversation, Message, ParticipantRole

logger = logging.getLogger(__name__)


def _format_conversation_history(messages: Sequence[Message]) -> str:
    """Formats a list of messages into a single string."""
    return "\n".join([f"{msg.sender.value.capitalize()}: {msg.content}" for msg in messages])


def _get_metadata(metadata_or_doc: dict | Document) -> dict:
    """Safely retrieve metadata if needed.

    A workaround for the fact that filter functions in different LangChain vector stores
    may require different metadata formats.
    """
    if isinstance(metadata_or_doc, Document):
        metadata: dict =  metadata_or_doc.metadata
        return metadata
    elif isinstance(metadata_or_doc, dict):
        return metadata_or_doc
    raise TypeError("metadata_or_doc must be either a Document or dict")


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

            outcome = conversation.outcome.name if conversation.outcome else None

            # The 'next message' becomes the metadata
            metadata = {
                "current_message_index": i,
                "has_next_message": bool(next_message),
                "current_message_role": current_message.sender.value,
                "current_message_timestamp": current_message.timestamp.isoformat(),
                "full_conversation": full_conversation,
                "outcome": outcome
            }

            if next_message:
                metadata["next_message_role"] = next_message.sender.value
                metadata["next_message_content"] = next_message.content
                metadata["next_message_timestamp"] = next_message.timestamp.isoformat()

            documents.append(Document(page_content=page_content, metadata=metadata))
    return documents


async def get_similar_finished_conversations(
        vector_store: VectorStore,
        conversation: Conversation,
        k: int
) -> list[tuple[Document, float]]:
    """Retrieve similar finished conversation examples from the vector store.

    Formats the current conversation as a query and searches for similar
    conversations in the vector store. Only completed conversations
    (has_next_message: False) are included, to retrieve finished conversations.

    Args:
        vector_store: The vector store to search for similar conversations
        conversation: Current conversation to find examples for
        k: Number of similar conversations to retrieve

    Returns:
        List of similar conversations as (Document, score) tuples, sorted by relevance
        and deduplicated by conversation
    """
    query = _format_conversation_history(conversation.messages)

    def filter_by_finished_conversation(metadata_or_doc: dict | Document) -> bool:
        """Filter function to check if the conversation is finished."""
        return _get_metadata(metadata_or_doc).get("has_next_message", False) is False

    # Retrieve similar conversations, filtering for finished conversations only
    retrieved_docs: list[tuple[Document, float]] = await vector_store.asimilarity_search_with_score(
        query=query,
        k=k,
        filter=filter_by_finished_conversation
    )

    # Sort by similarity score (highest first)
    retrieved_docs.sort(key=lambda x: x[1], reverse=True)

    return retrieved_docs


async def get_similar_examples_for_next_message_role(
    conversation_history: Sequence[Message],
    vector_store: VectorStore,
    k: int,
    target_role: ParticipantRole,
) -> list[Document]:
    """Retrieves examples from the vector store where the next message is from the specified role.
    
    This function finds conversations similar to the provided history where the subsequent
    message was authored by the specified role (e.g., AGENT, CUSTOMER). This allows for
    efficient RAG implementations across different participant types using a single index.
    
    Args:
        conversation_history: The current conversation history
        vector_store: Vector store containing the indexed conversations
        k: Number of examples to retrieve
        target_role: The role to filter results for (e.g., AGENT, CUSTOMER)
        
    Returns:
        List of relevant Document objects for the specified role
        
    Raises:
        ValueError: If not enough valid documents are retrieved
    """
    query = _format_conversation_history(conversation_history)

    def filter_by_matching_next_speaker(metadata_or_doc: dict | Document) -> bool:
        """Filter function to check if the next message role matches the target role."""
        role: str = _get_metadata(metadata_or_doc).get("next_message_role", "")
        return role == target_role.value
    
    # Filter for documents where the next_message_role matches the target role
    retrieved_docs: list[Document] = await vector_store.asimilarity_search(
        query=query,
        k=k,  # Use the exact k value requested
        filter=filter_by_matching_next_speaker
    )
    
    # Filter documents to ensure they have all required metadata
    valid_docs = [
        doc for doc in retrieved_docs
        if doc.metadata.get("next_message_content", "").strip()  # Ensure content is not empty
    ]
    
    return valid_docs

