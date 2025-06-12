
import logging
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from langchain_community.vectorstores import FAISS
from ..models import Message, ParticipantRole

logger = logging.getLogger(__name__)


def _conversations_to_langchain_documents(
    conversations: List[Dict[str, Any]],
) -> Tuple[List[Document], List[Document]]:
    """Converts raw conversations to Langchain Document objects.

    Args:
        conversations: List of conversation dictionaries.

    Returns:
        A tuple containing (customer_documents, agent_documents).
    """
    customer_documents: List[Document] = []
    agent_documents: List[Document] = []

    logger.info(f"Processing {len(conversations)} conversations into Langchain Documents.")

    for conv_idx, conv_data in enumerate(conversations):
        conversation_id = str(conv_data.get("id", f"conv_{conv_idx}"))
        messages_data = conv_data.get("messages", [])
        
        for msg_idx, msg_data in enumerate(messages_data):
            role_str = msg_data.get("role", "unknown").lower()
            content = msg_data.get("content", "")

            if not content:
                logger.debug(f"Skipping empty message {msg_idx} in {conversation_id}")
                continue

            metadata = {
                "conversation_id": conversation_id,
                "message_index": msg_idx,
                "role": role_str,
            }

            doc = Document(page_content=content, metadata=metadata)

            if role_str == ParticipantRole.CUSTOMER.value:
                customer_documents.append(doc)
            elif role_str == ParticipantRole.AGENT.value:
                agent_documents.append(doc)
            else:
                logger.warning(
                    f"Unknown role '{role_str}' in {conversation_id}, message {msg_idx}. "
                    f"Content: '{content[:50]}...'"
                )
            
    logger.info(
        f"Created {len(customer_documents)} customer Documents and "
        f"{len(agent_documents)} agent Documents."
    )
    return customer_documents, agent_documents


async def _langchain_documents_to_vector_store(
    docs: List[Document],
    openai_embeddings: OpenAIEmbeddings, 
) -> FAISS:
    """Creates a FaissVectorStore from a list of Langchain Document objects using OpenAI embeddings.

    Args:
        docs: List of Langchain Document objects.
        openai_embeddings: Initialized OpenAIEmbeddings instance.

    Returns:
        A populated FaissVectorStore instance.
    """
    if not docs:
        logger.info("No documents provided; creating an empty FAISS vector store.")
        # Create an empty FAISS index. Requires a dummy document and then clearing it,
        # or handling this scenario more gracefully if Langchain offers a direct way.
        # For now, let's create it with a placeholder and immediately have an empty store.
        # A truly empty FAISS store is tricky to initialize without any docs for from_documents.
        # Langchain's FAISS.from_texts with empty texts and a valid embedding function is one way.
        empty_store = FAISS.from_texts(texts=[""], embedding=openai_embeddings, metadatas=[{}], ids=["dummy"])
        empty_store.delete(["dummy"])
        return empty_store

    logger.info(f"Creating FAISS index from {len(docs)} documents using OpenAI embeddings. This may take a moment...")
    # Use FAISS.afrom_documents for asynchronous creation.
    try:
        faiss_index = await FAISS.afrom_documents(
            documents=docs,
            embedding=openai_embeddings
        )
    except Exception as e:
        logger.error(f"Error creating FAISS index with OpenAI embeddings: {e}")
        # Return an empty FAISS store on error
        logger.warning("Returning an empty FAISS vector store due to an error during index creation.")
        empty_store_on_error = FAISS.from_texts(texts=[""], embedding=openai_embeddings, metadatas=[{}], ids=["dummy"])
        empty_store_on_error.delete(["dummy"])
        return empty_store_on_error
    
    logger.info(f"Successfully created FAISS vector store with {len(docs)} documents.")
    return faiss_index


def convert_message_to_langchain(message: Message) -> BaseMessage:
    """Convert a Message object to its LangChain equivalent."""
    if message.sender == ParticipantRole.CUSTOMER:
        return HumanMessage(content=message.content)
    elif message.sender == ParticipantRole.AGENT:
        return AIMessage(content=message.content)
    else:
        # This case should ideally not be reached if roles are properly managed
        raise ValueError(f"Unknown sender role encountered: {message.sender}")


async def create_vector_stores_from_conversations(
    conversations: List[Dict[str, Any]],
    openai_api_key: str,
    openai_embedding_model_name: str = "text-embedding-ada-002",
) -> Tuple[FAISS, FAISS]:
    """Orchestrates the creation of vector stores from generic conversation data using OpenAI.

    Args:
        conversations: List of conversation dictionaries.
        openai_api_key: Your OpenAI API key.
        openai_embedding_model_name: Name of the OpenAI embedding model to use.

    Returns:
        A tuple (customer_vector_store, agent_vector_store).
    """
    if not conversations:
        raise ValueError("No conversations provided to create vector stores.")
    if not openai_api_key:
        raise ValueError("OpenAI API key is required.")

    logger.info(
        f"Starting vector store creation for {len(conversations)} conversations "
        f"using OpenAI model: {openai_embedding_model_name}."
    )

    # 1. Convert raw conversations to Langchain Document objects.
    # This step is synchronous and CPU-bound.
    customer_docs, agent_docs = _conversations_to_langchain_documents(conversations)

    # 2. Instantiate the OpenAI Embeddings model.
    try:
        openai_embeddings = OpenAIEmbeddings(
            api_key=SecretStr(openai_api_key), model=openai_embedding_model_name
        )
    except Exception as e:
        logger.error(f"Error initializing OpenAIEmbeddings: {e}")
        raise ValueError(f"Failed to initialize OpenAIEmbeddings: {e}") from e

    # 3. Create vector stores from the Langchain Documents.
    # These calls involve network I/O for OpenAI API and are run in threads.
    logger.info("Creating customer vector store...")
    customer_vector_store = await _langchain_documents_to_vector_store(
        customer_docs, openai_embeddings
    )
    logger.info("Creating agent vector store...")
    agent_vector_store = await _langchain_documents_to_vector_store(
        agent_docs, openai_embeddings
    )

    logger.info("Successfully created customer and agent vector stores using OpenAI.")
    return customer_vector_store, agent_vector_store


async def get_few_shot_examples_for_customer(
    agent_message: Message,
    customer_vector_store: FAISS,
    all_conversations: List[Dict[str, Any]],
    k: int = 3,
) -> List[Tuple[Dict, Dict]]:
    """Get few-shot examples for a customer based on the last agent message."""
    # 1. Find similar customer messages
    docs_and_scores = await customer_vector_store.asimilarity_search_with_score(
        agent_message.content, k=k
    )

    # 2. Find the preceding agent message for each customer message
    examples = []
    for doc, score in docs_and_scores:
        if "conversation_id" not in doc.metadata or "message_id" not in doc.metadata:
            logger.warning(
                f"Skipping document due to missing metadata: {doc.metadata}"
            )
            continue

        # Find the conversation
        conversation = next(
            (
                c
                for c in all_conversations
                if c["id"] == doc.metadata["conversation_id"]
            ),
            None,
        )
        if not conversation:
            continue

        # Find the customer message in the conversation
        customer_message_index = next(
            (
                i
                for i, msg in enumerate(conversation["messages"])
                if msg["id"] == doc.metadata["message_id"]
            ),
            None,
        )

        # Find the preceding agent message
        if customer_message_index is not None and customer_message_index > 0:
            preceding_message = conversation["messages"][customer_message_index - 1]
            if preceding_message["sender"] == "agent":
                examples.append((preceding_message, conversation["messages"][customer_message_index]))

    # 3. De-duplicate and return
    unique_examples = {}
    for agent_msg, customer_msg in examples:
        unique_examples[customer_msg["id"]] = (agent_msg, customer_msg)

    examples_to_return = list(unique_examples.values())
    logger.info(f"Found {len(examples_to_return)} few-shot examples for customer.")
    return examples_to_return


async def get_few_shot_examples_for_agent(
    last_customer_message_content: str,
    agent_vector_store: FAISS | None,
    all_conversations: List[Dict[str, Any]],
    k: int = 3,
) -> List[Tuple[Dict[str, str], Dict[str, str]]]:
    """
    Retrieves k few-shot examples from the agent_vector_store based on the last customer message.
    Each example consists of a preceding customer message and the retrieved agent message.

    Args:
        last_customer_message_content: The content of the last message from the customer.
        agent_vector_store: The FAISS vector store containing agent messages.
        all_conversations: The original list of all conversation data structures.
        k: The desired number of few-shot example pairs.

    Returns:
        A list of few-shot example pairs. Each pair is a list containing two dictionaries:
        [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
    """
    if not agent_vector_store or agent_vector_store.index.ntotal == 0:
        logger.info("Agent vector store is empty or not available. Cannot retrieve few-shot examples.")
        return []
    
    if not all_conversations:
        logger.warning("Full conversation data is not available. Cannot retrieve few-shot examples.")
        return []

    try:
        # Retrieve more documents than k initially, as some might not form valid pairs.
        retrieved_agent_docs: List[Document] = await agent_vector_store.asimilarity_search(
            query=last_customer_message_content,
            k=k * 2  # Heuristic: fetch more to increase chances of getting k valid pairs
        )
    except Exception as e:
        logger.error(f"Error during similarity search for few-shot examples: {e}")
        return []

    if not retrieved_agent_docs:
        logger.info(f"No relevant agent documents found for query: '{last_customer_message_content[:50]}...' for few-shot examples.")
        return []

    # Create a quick lookup for conversations by ID for efficient access
    conversations_by_id: Dict[str, Dict[str, Any]] = {
        str(conv.get("id")): conv for conv in all_conversations if conv.get("id")
    }

    few_shot_examples: List[Tuple[Dict[str, str], Dict[str, str]]] = []

    for agent_doc in retrieved_agent_docs:
        if len(few_shot_examples) >= k:
            break  # We have collected enough examples

        metadata = agent_doc.metadata
        conversation_id = metadata.get("conversation_id")
        agent_message_index = metadata.get("message_index")

        if conversation_id is None or agent_message_index is None:
            logger.debug(f"Retrieved agent document missing conversation_id or message_index: {metadata}")
            continue

        original_conv = conversations_by_id.get(str(conversation_id))
        if not original_conv:
            logger.debug(f"Could not find original conversation with ID: {conversation_id}")
            continue
        
        original_messages = original_conv.get("messages", [])

        # The agent message must have a preceding message
        if not (0 < agent_message_index < len(original_messages)):
            logger.debug(
                f"Agent message index {agent_message_index} is out of bounds or is the first message "
                f"in conversation {conversation_id}. Cannot get preceding customer message."
            )
            continue
            
        preceding_message_data = original_messages[agent_message_index - 1]
        preceding_message_role = preceding_message_data.get("role", "").lower()
        preceding_message_content = preceding_message_data.get("content", "")

        if preceding_message_role == ParticipantRole.CUSTOMER.value and preceding_message_content:
            example_pair = (
                {"role": "user", "content": preceding_message_content},
                {"role": "assistant", "content": agent_doc.page_content}
            )
            # Avoid adding duplicate example pairs if similarity search returns very similar agent messages
            # that map to the same preceding customer message.
            # This simple check might not be robust for all cases but helps for now.
            if example_pair not in few_shot_examples:
                 few_shot_examples.append(example_pair)
        else:
            logger.debug(
                f"Preceding message in {conversation_id} at index {agent_message_index -1} "
                f"is not a customer message or is empty. Role: '{preceding_message_role}', Content: '{preceding_message_content[:20]}...'. Skipping."
            )
            
    if not few_shot_examples:
        logger.info(f"Could not construct any valid few-shot examples for query: '{last_customer_message_content[:50]}...' after filtering.")

    return few_shot_examples[:k]  # Ensure we don't return more than k examples
