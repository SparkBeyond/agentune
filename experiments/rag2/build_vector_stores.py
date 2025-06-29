#!/usr/bin/env python3
"""
Build and save persistent FAISS vector store for RAG2 simulation.

This script creates a vector store from reference conversations and saves it to disk
for later use in simulations. For RAG2, we use a single vector store with metadata filtering.

Usage:
    python experiments/rag2/build_vector_stores.py
    
Requirements:
    - OpenAI API key set in environment (OPENAI_API_KEY)
    - langchain-community for FAISS support
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from conversation_simulator.models import Conversation
from conversation_simulator.rag import index_by_prefix

# Import common utilities from examples directory
sys.path.append(str(Path(__file__).parent.parent.parent / "examples"))
from common import load_sample_conversations

# Get module logger
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_EMBEDDINGS_MODEL = "text-embedding-3-small"
DEFAULT_DATA_FILE = "company_b_conversations.json"
DEFAULT_DATA_DIR = Path("data/sellence_headphones")
DEFAULT_OUTPUT_DIR = "vector_store"


async def build_persistent_vector_store(
    reference_conversations: List[Conversation],
    embeddings_model: Embeddings,
    output_dir: Path
) -> FAISS:
    """
    Build persistent FAISS vector store for RAG2 simulation.
    
    Args:
        reference_conversations: List of reference conversations
        embeddings_model: LangChain embeddings model instance
        output_dir: Directory to save the vector store
        
    Returns:
        FAISS vector store
    """
    print(f"Building persistent vector store from {len(reference_conversations)} conversations...")
    start_time = time.time()
    
    # Convert conversations to documents using the index_by_prefix module
    # No role filtering - RAG2 uses metadata filtering during retrieval
    print("Converting conversations to documents...")
    documents = index_by_prefix.conversations_to_langchain_documents(
        reference_conversations
    )
    
    print(f"Created {len(documents)} documents from conversations")
    
    # Create FAISS vector store
    print("Creating vector store...")
    vector_store = await FAISS.afrom_documents(
        documents=documents,
        embedding=embeddings_model
    )
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save vector store to disk
    print("Saving vector store to disk...")
    store_path = output_dir / "rag2_store"
    
    vector_store.save_local(str(store_path))
    
    elapsed_time = time.time() - start_time
    print(f"✅ Vector store built and saved successfully in {elapsed_time:.2f} seconds")
    print(f"   Store saved to: {store_path}")
    
    return vector_store


async def main() -> None:
    """Main function to build and save vector store."""
    # Validate OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Set paths
        script_dir = Path(__file__).parent
        data_file_path = script_dir / DEFAULT_DATA_DIR / DEFAULT_DATA_FILE
        output_dir = script_dir / DEFAULT_OUTPUT_DIR
        
        # Load conversation data
        print(f"Loading reference conversations from: {data_file_path}")
        reference_conversations = load_sample_conversations(data_file_path)
        
        print(f"Loaded {len(reference_conversations)} conversations")
        
        # Initialize embeddings model
        print(f"Initializing embeddings model: {DEFAULT_EMBEDDINGS_MODEL}")
        embeddings_model = OpenAIEmbeddings(model=DEFAULT_EMBEDDINGS_MODEL)
        
        # Build and save vector store
        await build_persistent_vector_store(
            reference_conversations=reference_conversations,
            embeddings_model=embeddings_model,
            output_dir=output_dir
        )
        
        print("\n🎉 Vector store creation completed!")
        print("You can now run simulations using: python experiments/rag2/run_simulation.py")
        
    except Exception as e:
        logger.error(f"Vector store creation failed: {e}")
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
