
import logging
from typing import List, Tuple, Type

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings

from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from ..models import Conversation, Message, ParticipantRole

logger = logging.getLogger(__name__)


def _format_conversation_history(messages: List[Message]) -> str:
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
    conversation_history: List[Message],
    vector_store: VectorStore | None,
    k: int,
    target_role: ParticipantRole,
    role_name_for_logs: str,
) -> List[Document]:
    """Retrieves k relevant documents for a given role from a vector store."""
    if not vector_store:
        logger.info(
            f"{role_name_for_logs.capitalize()} vector store is not available. "
            "Cannot retrieve few-shot examples."
        )
        return []

    query = _format_conversation_history(conversation_history)

    try:
        retrieved_docs: List[Document] = await vector_store.asimilarity_search(
            query=query, k=k
        )
    except Exception as e:
        logger.error(
            f"Error during similarity search for {role_name_for_logs} few-shot examples: {e}"
        )
        return []

    if not retrieved_docs:
        logger.info(
            f"No relevant {role_name_for_logs} documents found for query: "
            f"'{query[:100]}...' for few-shot examples."
        )
        return []

    valid_docs: List[Document] = []
    for doc in retrieved_docs:
        try:
            # Ensure essential metadata is present and role is correct
            if doc.metadata.get("role") == target_role.value:
                _ = doc.metadata["content"]  # Check for presence
                _ = doc.metadata["timestamp"]  # Check for presence
                valid_docs.append(doc)
            else:
                logger.warning(
                    f"Document from {role_name_for_logs}_vector_store has unexpected role '"
                    f"{doc.metadata.get('role')}'. Expected '{target_role.value}'. Skipping."
                )
        except KeyError as e:
            logger.error(
                f"Document missing essential metadata ('role', 'content', or 'timestamp'): "
                f"{doc.metadata}. Error: {e}. Skipping."
            )
            continue
        except (TypeError, ValueError) as e:
            logger.error(
                f"Error processing document metadata: {doc.metadata}. Error: {e}. Skipping."
            )
            continue

    logger.info(
        f"Retrieved {len(valid_docs)} valid documents for {role_name_for_logs} few-shot examples."
    )
    # The asimilarity_search already respects k, so no need to slice with [:k] here.
    return valid_docs


async def get_few_shot_examples_for_customer(
    conversation_history: List[Message],
    customer_vector_store: VectorStore | None, 
    k: int = 3,
) -> List[Document]:
    """Retrieves k relevant documents for customer few-shot examples."""
    return await _get_few_shot_examples(
        conversation_history=conversation_history,
        vector_store=customer_vector_store,
        k=k,
        target_role=ParticipantRole.CUSTOMER,
        role_name_for_logs="customer",
    )


async def get_few_shot_examples_for_agent(
    conversation_history: List[Message],
    agent_vector_store: VectorStore | None,
    k: int = 3,
) -> List[Document]:
    """
    Retrieves k relevant documents for agent few-shot examples.

    Args:
        conversation_history: The history of the current conversation.
        agent_vector_store: The vector store containing agent messages.
        k: The desired number of few-shot examples.

    Returns:
        A list of `Document` objects, where each document's page_content is the history
        and metadata contains the agent's response.
    """
    return await _get_few_shot_examples(
        conversation_history=conversation_history,
        vector_store=agent_vector_store,
        k=k,
        target_role=ParticipantRole.AGENT,
        role_name_for_logs="agent",
    )
