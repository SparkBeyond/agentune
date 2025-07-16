
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
        metadata: dict = metadata_or_doc.metadata
        return metadata
    elif isinstance(metadata_or_doc, dict):
        return metadata_or_doc
    raise TypeError("metadata_or_doc must be either a Document or dict")


def conversations_to_langchain_documents(
    conversations: list[Conversation]
) -> list[Document]:
    """
    Converts a list of conversations into a list of LangChain documents,
    where the content is the conversation history and the metadata is the current message index.
    """
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


async def get_few_shot_examples(
    conversation_history: Sequence[Message],
    vector_store: VectorStore,
    k: int
) -> list[tuple[Document, float]]:
    """Retrieves k relevant documents for a given role of the current last message."""
    if not conversation_history:
        return []

    current_message_role = conversation_history[-1].sender
    query = _format_conversation_history(conversation_history)

    def role_filter_function(document: Document) -> bool:
        """
        This function acts as the filter. It checks if a document's role
        matches the role of the current speaker.
        """
        # It uses your _get_metadata helper
        metadata = _get_metadata(document)
        # It has access to 'current_message_role' from the outer scope
        return metadata.get("current_message_role") == current_message_role.value

    retrieved_docs: list[tuple[Document, float]] = await vector_store.asimilarity_search_with_score(
        query=query, k=k,
        filter=role_filter_function
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

    logger.debug(f"Retrieved {len(retrieved_docs)} documents, deduplicated to {len(unique_docs)}.")

    return unique_docs


def format_examples(
        examples: list[tuple[Document, float]],
) -> str:
    """
    Formats the retrieved few-shot example Documents into a list of LangChain messages.
    Each Document's page_content (history, typically customer's turn) becomes a HumanMessage,
    and the metadata (next agent message) becomes an embedded agent response in the conversation history.
    """
    conversations: list[Document] = [doc[0] for doc in examples]
    formatted_messages: list[str] = []

    def format_single_conversation(ind, doc) -> str:
        """Formats a single conversation Document into a LangChain message."""
        try:
            # Extract metadata from the document
            metadata = doc.metadata
            # Validate essential keys before creating Message object
            if "full_conversation" not in metadata:
                logger.warning(f"Skipping example due to missing 'full_conversation' metadata: {metadata}")
                return ""

            full_conversation = str(metadata["full_conversation"])
            formatted_conv = full_conversation  # Default to the original conversation

            # 2. Conditional Logic: Only try to highlight the response if 'next_message_content' exists.
            if "next_message_content" in metadata:
                participant_response = str(metadata["next_message_content"])
                role_name = metadata["next_message_role"].capitalize()
                participant_turn_str = f"{role_name}: {participant_response}"

                if participant_turn_str in full_conversation:
                    # If the key exists, perform the replacement as before
                    formatted_conv = full_conversation.replace(
                        participant_turn_str, f"**Next {role_name} response**: {participant_response}",
                        1  # Only replace the first occurrence
                    )
                else:
                    logger.warning(
                        f"Could not find {role_name} turn '{participant_response}' in full conversation."
                    )

            # Add indexing for the examples
            return f"Example {ind + 1}:\n\n{formatted_conv}"
        except Exception as e:
            logger.error(f"Error formatting conversation example {ind + 1}: {e}")
            return ""

    for index, document in enumerate(conversations):
        formatted_messages.append(format_single_conversation(index, document))

    return "\n\n".join(formatted_messages)