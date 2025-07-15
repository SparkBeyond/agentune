
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document

from langchain_core.vectorstores import VectorStore
from ..models import Conversation, Message, ParticipantRole
from ..util.structure import converter

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    role: str
    content: str
    is_target: bool = False


def format_conversation(messages: Sequence[Message]) -> str:
    """Formats a list of messages into a single string."""
    return "\n".join([f"{msg.sender.value.capitalize()}: {msg.content}" for msg in messages])


def format_conversation_with_highlight(messages: Sequence[Message], current_index: int) -> str:
    """Formats a conversation with the next message after current_index highlighted.
    
    Args:
        messages: Sequence of messages to format
        current_index: Index of the current message (next message will be highlighted)
        
    Returns:
        Formatted conversation string with next message highlighted
    """
    formatted_lines = []
    highlight_index = current_index + 1
    
    for i, msg in enumerate(messages):
        if i == highlight_index and highlight_index < len(messages):
            highlight_label = f"Current {msg.sender.value.capitalize()} response"
            formatted_lines.append(f"**{highlight_label}**: {msg.content}")
        else:
            formatted_lines.append(f"{msg.sender.value.capitalize()}: {msg.content}")
    return "\n".join(formatted_lines)


def format_highlighted_example(doc: Document) -> str:
    """Format a single example document with proper highlighting.
    
    Args:
        doc: Document containing conversation metadata
        
    Returns:
        Formatted conversation string with the next message highlighted
    """
    metadata = doc.metadata
    
    # Deserialize the structured messages
    messages_data = metadata["full_conversation"]
    messages = converter.structure(messages_data, list[Message])
    
    # Get the current message index and use it to highlight the next message
    current_index = metadata["current_message_index"]
    
    return format_conversation_with_highlight(messages, current_index)


def _format_conversation_history(messages: Sequence[Message]) -> str:
    """Deprecated: Use format_conversation instead."""
    return format_conversation(messages)


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
    where the content is the conversation history and the metadata contains structured message data.
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
            page_content = format_conversation(history_messages)

            outcome = conversation.outcome.name if conversation.outcome else None

            # The 'next message' becomes the metadata
            metadata = {
                "conversation_hash": hash(conversation.messages),
                "current_message_index": i,
                "has_next_message": bool(next_message),
                "current_message_role": current_message.sender.value,
                "current_message_timestamp": current_message.timestamp.isoformat(),
                "full_conversation": converter.unstructure(conversation.messages),
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


def parse_conversation_turns(
    conversation: str, 
    target_response: str, 
    target_role: ParticipantRole
) -> list[ConversationTurn]:
    """Parse conversation into structured turns, highlighting target role's specific response.
    
    Args:
        conversation: Full conversation string with "Role: content" format
        target_response: The specific response to highlight as the target
        target_role: The role whose response should be highlighted
        
    Returns:
        List of ConversationTurn objects with target response marked
    """
    
    # Define the expected role prefixes
    CUSTOMER_PREFIX = "Customer: "
    AGENT_PREFIX = "Agent: "
    
    # Map roles to their prefixes
    role_prefixes = {
        ParticipantRole.CUSTOMER: CUSTOMER_PREFIX,
        ParticipantRole.AGENT: AGENT_PREFIX
    }
    
    turns = []
    target_found = False
    
    for line in conversation.split('\n'):  # This could separate existing messages.
        line = line.strip()
        if not line:
            continue
        
        # Try to parse the line for each known role
        parsed_turn = None
        
        for role, prefix in role_prefixes.items():
            if line.startswith(prefix):
                content = line[len(prefix):]
                
                # Mark as target if: 1) it's the role we're looking for, 
                # 2) the content exactly matches our target response, and 
                # 3) we haven't already found a target (prevents duplicate highlighting)
                is_target = (
                    role == target_role and 
                    content == target_response and 
                    not target_found
                )
                if is_target:
                    target_found = True
                
                parsed_turn = ConversationTurn(role.value.capitalize(), content, is_target)
                break
        
        if parsed_turn:
            turns.append(parsed_turn)
        else:
            # Handle unexpected format
            logger.warning(f"Unexpected line format in conversation: {line}")
    
    return turns


async def get_few_shot_examples(
    conversation_history: Sequence[Message],
    vector_store: VectorStore,
    k: int
) -> list[tuple[Document, float]]:
    """Retrieves k relevant documents for a given role of the current last message."""

    current_message_role = conversation_history[-1].sender
    query = format_conversation(conversation_history)

    def role_filter_function(doc):
        """
        This function acts as the filter. It checks if a document's role
        matches the role of the current speaker.
        """
        # It uses your _get_metadata helper
        metadata = _get_metadata(doc)
        # It has access to 'current_message_role' from the outer scope
        return metadata.get("current_message_role") == current_message_role.value

    retrieved_docs: list[tuple[Document, float]] = await vector_store.asimilarity_search_with_score(
        query=query, k=k,
        filter=role_filter_function
    )

    # Sort retrieved docs by score
    retrieved_docs.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate documents coming from the same conversation, by comparing the conversation_hash metadata
    unique_docs = []
    seen_conversations = set()
    for doc, score in retrieved_docs:
        conversation_hash = doc.metadata.get("conversation_hash")
        if conversation_hash not in seen_conversations:
            unique_docs.append((doc, score))
            seen_conversations.add(conversation_hash)

    logger.debug(f"Retrieved {len(retrieved_docs)} documents, deduplicated to {len(unique_docs)}.")

    return unique_docs


# _format_examples function removed - participants now handle their own example formatting