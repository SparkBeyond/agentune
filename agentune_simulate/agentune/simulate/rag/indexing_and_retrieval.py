import logging
from collections.abc import Sequence

from langchain_core.documents import Document

from langchain_core.vectorstores import VectorStore
from ..models import Conversation, Message
from ..util.structure import converter
from .filtered_retriever import VectorStoreSearcher

logger = logging.getLogger(__name__)


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
        if i == current_index:  # Highlight the last message
            highlight_label = f"Last {msg.sender.value.capitalize()} message"
            formatted_lines.append(f"**{highlight_label}**: {msg.content}")
        elif i == highlight_index and highlight_index < len(messages):
            highlight_label = f"Current {msg.sender.value.capitalize()} response"
            formatted_lines.append(f"**{highlight_label}**: {msg.content}")
        else:
            formatted_lines.append(f"{msg.sender.value.capitalize()}: {msg.content}")
    return "\n".join(formatted_lines)


def format_examples(examples: list[tuple[Document, float]]) -> str:
    """Format few-shot examples for the customer prompt."""
    if not examples:
        return "No examples available."

    formatted_examples = []

    for index, (doc, score) in enumerate(examples):
        formatted_conversation = format_highlighted_example(doc)
        formatted_examples.append(f"Example {index + 1}:")
        formatted_examples.append(formatted_conversation)

    return "\n\n".join(formatted_examples)


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
    messages = converter.loads(messages_data, list[Message])

    # Get the current message index and use it to highlight the next message
    current_index = metadata["current_message_index"]

    return format_conversation_with_highlight(messages, current_index)


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
            next_message: Message | None = conversation.messages[i + 1] if i + 1 < len(conversation.messages) else None

            history_messages: list[Message] = list(conversation.messages[:i + 1])
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
                "full_conversation": converter.dumps(conversation.messages),
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
    query = format_conversation(conversation.messages)

    # Use the new searcher for consistent filtering across vector stores
    searcher = VectorStoreSearcher.create(vector_store)
    retrieved_docs = await searcher.similarity_search_with_filter(
        query=query,
        k=k,
        filter_dict={"has_next_message": False}
    )

    # Sort by similarity score (highest first)
    retrieved_docs.sort(key=lambda x: x[1], reverse=True)

    return retrieved_docs


async def get_few_shot_examples(
    conversation_history: Sequence[Message],
    vector_store: VectorStore,
    k: int
) -> list[tuple[Document, float]]:
    """Retrieves k relevant documents for a given role of the current last message."""
    if not conversation_history:
        return []

    current_message_role = conversation_history[-1].sender
    query = format_conversation(conversation_history)

    # Use the new searcher for consistent filtering across vector stores
    searcher = VectorStoreSearcher.create(vector_store)
    retrieved_docs = await searcher.similarity_search_with_filter(
        query=query,
        k=k,
        filter_dict={"current_message_role": current_message_role.value}
    )

    # Sort retrieved docs by score
    retrieved_docs.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate documents coming from the same conversation, by comparing the conversation_hash metadata
    unique_docs = []
    seen_conversations = set()
    for doc, score in retrieved_docs:
        conversation_hash = doc.metadata.get("conversation_hash")
        if conversation_hash not in seen_conversations:  # Verified manually that the hash is consistent
            unique_docs.append((doc, score))
            seen_conversations.add(conversation_hash)

    logger.debug(f"Retrieved {len(retrieved_docs)} documents, deduplicated to {len(unique_docs)}.")

    return unique_docs
