
import logging
from typing import List, Tuple, Type, Sequence

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings

from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from ..models import Conversation, Message, ParticipantRole

logger = logging.getLogger(__name__)


def _format_conversation_history(messages: Sequence[Message]) -> str:
    """Formats a list of messages into a single string."""
    return "\n".join([f"{msg.sender.value.capitalize()}: {msg.content}" for msg in messages])

def _conversations_to_langchain_documents(
    conversations: List[Conversation],
) -> Tuple[List[Document], List[Document]]:
    """Converts raw conversations to Langchain Document objects.

    Args:
        conversations: List of Conversation objects.

    Returns:
        A tuple containing (customer_documents, agent_documents).
    """
    customer_documents: List[Document] = []
    agent_documents: List[Document] = []

    logger.info(f"Processing {len(conversations)} conversations into Langchain Documents.")

    for conversation in conversations:
        if len(conversation.messages) < 2:
            continue # Need at least one message for history and one for the 'next message'

        for i in range(1, len(conversation.messages)):
            history_messages: List[Message] = list(conversation.messages[:i])
            next_message: Message = conversation.messages[i]

            if not next_message.content:
                logger.debug(f"Skipping empty next_message at index {i} in conversation starting with: {conversation.messages[0].content[:50]}...")
                continue

            # The history becomes the content to be embedded
            page_content = _format_conversation_history(history_messages)

            # The 'next message' becomes the metadata
            metadata = {
                "message_index": i,
                "role": next_message.sender.value,
            }
            if next_message.content:
                metadata["content"] = next_message.content  # Store the response content
            metadata["timestamp"] = next_message.timestamp.isoformat()

            doc = Document(page_content=page_content, metadata=metadata)

            # Assign document to a store based on the role of the 'next message'
            if next_message.sender == ParticipantRole.CUSTOMER:
                customer_documents.append(doc)
            elif next_message.sender == ParticipantRole.AGENT:
                agent_documents.append(doc)

    logger.info(
        f"Created {len(customer_documents)} customer Documents and "
        f"{len(agent_documents)} agent Documents."
    )
    return customer_documents, agent_documents


async def _langchain_documents_to_vector_store(
    docs: List[Document],
    openai_embeddings: OpenAIEmbeddings,
    vector_store_class: Type[VectorStore] = FAISS,
) -> VectorStore:
    """Creates a vector store from a list of Langchain Document objects.

    Args:
        docs: List of Langchain Document objects.
        openai_embeddings: Initialized OpenAIEmbeddings instance.
        vector_store_class: The vector store class to use (e.g., FAISS).

    Returns:
        A populated VectorStore instance.
    """
    if not docs:
        logger.warning(
            f"No documents provided to create a vector store. "
            f"Returning a {vector_store_class.__name__} store with one dummy document."
        )
        dummy_doc = Document(page_content="dummy")
        return await vector_store_class.afrom_documents([dummy_doc], openai_embeddings)

    logger.info(f"Creating {vector_store_class.__name__} index from {len(docs)} documents using OpenAI embeddings. This may take a moment...")
    try:
        index = await vector_store_class.afrom_documents(documents=docs, embedding=openai_embeddings)
        logger.info(f"Successfully created {vector_store_class.__name__} vector store with {len(docs)} documents.")
        return index
    except Exception as e:
        logger.error(f"Error creating {vector_store_class.__name__} index with OpenAI embeddings: {e}")
        logger.warning(f"Returning a {vector_store_class.__name__} vector store with one dummy document due to an error.")
        dummy_doc = Document(page_content="dummy")
        return await vector_store_class.afrom_documents([dummy_doc], openai_embeddings)


def convert_message_to_langchain(message: Message) -> BaseMessage:
    """Convert a Message object to its LangChain equivalent."""
    if message.sender == ParticipantRole.CUSTOMER:
        return HumanMessage(content=message.content)
    elif message.sender == ParticipantRole.AGENT:
        return AIMessage(content=message.content)
    else:
        raise ValueError(f"Unknown sender role encountered: {message.sender}")


async def create_vector_stores_from_conversations(
    conversations: List[Conversation],
    openai_embedding_model_name: str = "text-embedding-ada-002",
    vector_store_class: Type[VectorStore] = FAISS,
) -> Tuple[VectorStore, VectorStore]:
    """Orchestrates the creation of vector stores from generic conversation data.

    Args:
        conversations: List of conversation dictionaries.
        openai_embedding_model_name: Name of the OpenAI embedding model to use.
        vector_store_class: The vector store class to use (e.g., FAISS).

    Returns:
        A tuple (customer_vector_store, agent_vector_store).
    """
    logger.info(
        f"Starting vector store creation for {len(conversations)} conversations "
        f"using OpenAI model: {openai_embedding_model_name} and "
        f"vector store: {vector_store_class.__name__}."
    )
    openai_embeddings = OpenAIEmbeddings(model=openai_embedding_model_name)

    if not conversations:
        logger.warning("No conversations provided. Returning empty vector stores.")
        empty_store = await _langchain_documents_to_vector_store(
            [], openai_embeddings, vector_store_class=vector_store_class
        )
        return empty_store, empty_store

    customer_docs, agent_docs = _conversations_to_langchain_documents(conversations)

    logger.info(f"Creating customer vector store ({vector_store_class.__name__})...")
    customer_vector_store = await _langchain_documents_to_vector_store(
        customer_docs, openai_embeddings, vector_store_class=vector_store_class
    )
    logger.info(f"Creating agent vector store ({vector_store_class.__name__})...")
    agent_vector_store = await _langchain_documents_to_vector_store(
        agent_docs, openai_embeddings, vector_store_class=vector_store_class
    )

    logger.info(f"Successfully created customer and agent vector stores using {vector_store_class.__name__}.")
    return customer_vector_store, agent_vector_store


async def _get_few_shot_examples(
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
