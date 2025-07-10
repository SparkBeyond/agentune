import logging
from collections.abc import Sequence

from langchain_core.documents import Document

from langchain_core.vectorstores import VectorStore
from ..models import Conversation, Message, ParticipantRole

logger = logging.getLogger(__name__)


def _format_conversation_history(messages: Sequence[Message]) -> str:
    """Formats a list of messages into a single string."""
    return "\n".join([f"{msg.sender.value.capitalize()}: {msg.content}" for msg in messages])

def _extract_focused_part_of_conversation(
    messages: Sequence[Message],
    current_index: int,
    focus_size: int,
) -> str:
    """Return a formatted view on a conversation centred on ``current_index``.

    The window includes ``focus_size`` messages *before* and *after* the current
    message (clamped to the conversation bounds). The format shows the conversation
    context, who should respond next, and what the actual response was.

    Examples:

        Example 1:
        *Last few messages until the latest message*
        ...
        CUSTOMER: Music
        AGENT: What kind of music do you like?
        CUSTOMER: I like rock music, especially classic rock.

        *Next to respond*: AGENT
        *Response*: "That's great! Classic rock has some amazing bands. Do you have a favorite band?"

        Example 2:
        *Last few messages until the latest message*
        AGENT: Hi there! How can I help you today?
        CUSTOMER: Looking to buy a thing.

        *Next to respond*: AGENT
        *Response*: "Sure, I can help with that. What kind of thing are you looking for?"

        Example 3:
        *Last few messages until the latest message*
        ...
        AGENT: Is there anything else I can assist you with?
        CUSTOMER: No, that's all for now. Thank you!

        *Next to respond*: AGENT
        *Response*: "You're welcome! Have a great day!"

        Example 4:
        AGENT: Hi, how can I help you today?  

        *Next to respond*: None, the conversation has ended without additional messages.
    """

    # ---- Input validation -------------------------------------------------
    if not messages:
        raise ValueError("messages is empty")

    if focus_size < 0:
        raise ValueError("focus_size must be non‑negative")

    if current_index < 0 or current_index >= len(messages):
        raise ValueError("current_index is out of range")

    # ---- Determine window for context messages -----------------------------
    start_index = max(0, current_index - focus_size)
    end_index = current_index + 1  # Include up to current message

    context_messages: Sequence[Message] = messages[start_index:end_index]

    # ---- Build formatted output -------------------------------------------
    lines: list[str] = []
    
    # Add header
    lines.append("*Last few messages until the latest message*")
    
    # Add ellipsis if we're not showing from the beginning
    if start_index > 0:
        lines.append("...")
    
    # Add context messages
    for msg in context_messages:
        lines.append(f"{msg.sender.value.upper()}: {msg.content}")
    
    # Add empty line before next to respond
    lines.append("")
    
    # Determine who should respond next and what the response was
    next_message_index = current_index + 1
    if next_message_index < len(messages):
        next_message = messages[next_message_index]
        lines.append(f"*Next to respond*: {next_message.sender.value.upper()}")
        lines.append(f'*Response*: "{next_message.content}"')
    else:
        lines.append("*Next to respond*: None, the conversation has ended without additional messages.")

    return "\n".join(lines)


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
            focused_conversation_part = _extract_focused_part_of_conversation(
                conversation.messages, i, focus_size=2
            )

            outcome = conversation.outcome.name if conversation.outcome else None

            # The 'next message' becomes the metadata
            metadata = {
                "current_message_index": i,
                "has_next_message": bool(next_message),
                "current_message_role": current_message.sender.value,
                "current_message_timestamp": current_message.timestamp.isoformat(),
                "full_conversation": full_conversation,
                "focused_conversation_part": focused_conversation_part,
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
        # filter=filter_by_finished_conversation
        filter = {"has_next_message": False}
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
        # filter=filter_by_matching_next_speaker
        filter={"next_message_role": target_role.value}  # Use a dictionary filter for LangChain
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

    logger.debug(f"Retrieved {len(retrieved_docs)} documents, deduplicated to {len(unique_docs)}.")

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

def _parse_conversation_from_full_text(full_conversation: str) -> list[Message]:
    """Parse a full conversation string back into Message objects.
    
    Args:
        full_conversation: String formatted as "ROLE: content\nROLE: content\n..."
        
    Returns:
        List of Message objects
    """
    from datetime import datetime
    
    messages = []
    lines = full_conversation.strip().split('\n')
    
    for line in lines:
        if ':' not in line:
            continue
            
        role_str, content = line.split(':', 1)
        role_str = role_str.strip().upper()
        content = content.strip()
        
        try:
            role = ParticipantRole(role_str.lower())
            message = Message(
                sender=role,
                content=content,
                timestamp=datetime.now()  # Placeholder timestamp
            )
            messages.append(message)
        except ValueError:
            # Skip invalid role lines
            continue
    
    return messages


def format_focused_example(doc: Document, example_num: int, focus_size: int = 2) -> str:
    """Format a document as a focused example conversation.
    
    Args:
        doc: Document containing conversation metadata
        example_num: Number for labeling the example
        focus_size: Number of messages to include before and after current message
        
    Returns:
        Formatted focused conversation example
    """
    current_index = doc.metadata.get('current_message_index', 0)
    full_conversation = doc.metadata.get('full_conversation', '')
    
    if not full_conversation:
        return f"Example conversation {example_num}:\n[No conversation data available]"
    
    # Parse the conversation back into Message objects
    messages = _parse_conversation_from_full_text(full_conversation)
    
    if not messages or current_index >= len(messages):
        return f"Example conversation {example_num}:\n{full_conversation}"
    
    # Extract focused part
    focused_part = _extract_focused_part_of_conversation(messages, current_index, focus_size)
    
    return f"Example conversation {example_num}:\n{focused_part}"
