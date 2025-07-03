#!/usr/bin/env python3
"""
Run RAG2 simulation using pre-built persistent vector store.

This script loads previously created FAISS vector store from disk and runs
simulations with configurable parameters. For RAG2, we use a single vector store
with metadata filtering during retrieval.

Usage:
    python experiments/rag2/run_simulation.py [--simulations N]
    
Requirements:
    - Vector store must be built first using build_vector_stores.py
    - OpenAI API key set in environment (OPENAI_API_KEY)
    - langchain-community for FAISS support
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

# Import common utilities from examples directory
sys.path.append(str(Path(__file__).parent.parent.parent / "examples"))
from common import load_sample_conversations, converter

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from conversation_simulator.models import Conversation, Outcomes
from conversation_simulator.models.results import SimulationSessionResult
from conversation_simulator.participants.agent.rag2 import Rag2AgentFactory
from conversation_simulator.participants.customer.rag2 import RagCustomerFactory
from conversation_simulator.simulation.session_builder import SimulationSessionBuilder

# Get module logger
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_EMBEDDINGS_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o"
DEFAULT_VECTOR_STORE_DIR = "vector_store"
DEFAULT_NUM_SIMULATIONS = 5  # Lower default for initial testing


def load_persistent_vector_store(
    vector_store_dir: Path,
    embeddings_model: Embeddings
) -> FAISS:
    """
    Load persistent FAISS vector store from disk.
    
    Args:
        vector_store_dir: Directory containing the saved vector store
        embeddings_model: LangChain embeddings model instance (must match the one used during creation)
        
    Returns:
        FAISS vector store
        
    Raises:
        FileNotFoundError: If vector store doesn't exist
        ValueError: If vector store is corrupted
    """
    store_path = vector_store_dir / "rag2_store"
    
    # Check if vector store exists
    if not store_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {store_path}. "
            "Please run build_vector_stores.py first."
        )
    
    print("Loading vector store from disk...")
    start_time = time.time()
    
    try:
        # Load FAISS vector store
        vector_store = FAISS.load_local(
            str(store_path),
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"✅ Vector store loaded successfully in {elapsed_time:.2f} seconds")
        
        return vector_store
        
    except Exception as e:
        raise ValueError(f"Failed to load vector store: {e}") from e


async def run_simulation_with_persistent_store(
    vector_store: FAISS,
    chat_model: BaseChatModel,
    reference_conversations: list[Conversation],
    number_of_simulations: int = DEFAULT_NUM_SIMULATIONS
) -> SimulationSessionResult:
    """
    Run RAG2 simulation using pre-loaded vector store.
    
    Args:
        vector_store: Pre-loaded vector store (shared between agent and customer)
        chat_model: LangChain chat model instance
        reference_conversations: List of reference conversations
        number_of_simulations: Number of simulations to run
        
    Returns:
        SimulationSessionResult object containing the simulation results
    """
    rand = random.Random(42)
    print(f"Setting up simulation with {number_of_simulations} conversations...")
    start_time = time.time()
    
    # Create RAG2 participant factories with shared vector store
    agent_factory = Rag2AgentFactory(
        model=chat_model,
        agent_vector_store=vector_store
    )
    
    customer_factory = RagCustomerFactory(
        model=chat_model,
        customer_vector_store=vector_store,
        seed=42  # Use a fixed seed for deterministic results
    )
    
    # Extract outcomes if available in the reference data
    outcomes = extract_outcomes_from_conversations(reference_conversations)
    
    # Build simulation session
    session = SimulationSessionBuilder(
        agent_factory=agent_factory,
        customer_factory=customer_factory,
        outcomes=outcomes,
        chat_model=chat_model,  # Add chat_model as required by the API
        session_description="RAG2 simulation using persistent vector store",
        max_messages=40,  # Reasonable limit for the example
    ).build()
    
    # Run simulation with reference conversations
    if len(reference_conversations) > number_of_simulations:
        base_conversations = rand.sample(reference_conversations, number_of_simulations)
        print(f"Using {number_of_simulations} conversations as simulation base")
    else:
        base_conversations = reference_conversations
    
    result = await session.run_simulation(base_conversations)
    
    elapsed_time = time.time() - start_time
    print(f"✅ Simulation completed in {elapsed_time:.2f} seconds")
    print(f"Generated {len(result.simulated_conversations)} conversations")
    
    return result


def extract_outcomes_from_conversations(conversations: list[Conversation]) -> Outcomes:
    """
    Extract unique outcomes from reference conversations.
    
    Args:
        conversations: List of reference conversations
        
    Returns:
        Outcomes object containing unique outcome categories
        
    Raises:
        ValueError: If no outcomes are found in the conversations
    """
    unique_outcomes = {}
    
    for conversation in conversations:
        if conversation.outcome:
            outcome_name = conversation.outcome.name
            if outcome_name not in unique_outcomes:
                unique_outcomes[outcome_name] = conversation.outcome
    
    if not unique_outcomes:
        print("Warning: No outcomes found in conversations. Using empty outcomes.")
        return Outcomes(outcomes=())
    
    outcomes_tuple = tuple(unique_outcomes.values())
    print(f"Extracted {len(outcomes_tuple)} unique outcomes: {[o.name for o in outcomes_tuple]}")
    return Outcomes(outcomes=outcomes_tuple)


def print_simple_summary(result: SimulationSessionResult) -> None:
    """
    Print a simple summary of simulation results.
    
    Args:
        result: SimulationSessionResult from simulation run
    """
    print("\n" + "=" * 50)
    print("SIMULATION RESULTS")
    print("=" * 50)
    
    print("\nSummary of Results:")
    print(f"Original conversations: {len(result.original_conversations)}")
    print(f"Simulated conversations: {len(result.simulated_conversations)}")


async def main() -> None:
    """
    Main function to run simulation with persistent vector store.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run RAG2 simulation with persistent vector store")
    parser.add_argument(
        "--simulations", 
        type=int, 
        default=DEFAULT_NUM_SIMULATIONS,
        help=f"Number of simulations to run (default: {DEFAULT_NUM_SIMULATIONS})"
    )
    args = parser.parse_args()
    
    # Validate OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Set paths
        script_dir = Path(__file__).parent
        vector_store_dir = script_dir / DEFAULT_VECTOR_STORE_DIR
        data_dir = script_dir / "data/sellence_headphones"
        data_file = data_dir / "company_b_conversations.json"
        
        # Load reference conversations
        print(f"Loading reference conversations from: {data_file}")
        reference_conversations = load_sample_conversations(data_file)
        
        print(f"Loaded {len(reference_conversations)} conversations")

        # Configure LangChain logging
        from langchain.callbacks.tracers.logging import LoggingCallbackHandler
        langchain_logger = logging.getLogger("langchain")
        langchain_logger.setLevel(logging.INFO)

        # Attach LangChain’s built-in tracer callback
        tracer = LoggingCallbackHandler(logger=langchain_logger, log_level=logging.INFO)  # :contentReference[oaicite:0]{index=0}
        
        # Initialize models
        embeddings_model = OpenAIEmbeddings(model=DEFAULT_EMBEDDINGS_MODEL)
        chat_model = ChatOpenAI(model=DEFAULT_CHAT_MODEL, temperature=0.0, callbacks=[tracer])
        # chat_model = ChatOpenAI(model="o3")
        
        # Load persistent vector store
        vector_store = load_persistent_vector_store(
            vector_store_dir=vector_store_dir,
            embeddings_model=embeddings_model
        )
        
        # Run simulation
        result = await run_simulation_with_persistent_store(
            vector_store=vector_store,
            chat_model=chat_model,
            reference_conversations=reference_conversations,
            number_of_simulations=args.simulations
        )
        
        # Display results
        print("\n" + "="*50)
        print("SIMULATION RESULTS")
        print("="*50)
        print_simple_summary(result)
        
        # Save results
        output_file = script_dir / "rag2_simulation_results.json"
        
        # Convert to dict for JSON serialization using cattrs
        result_dict = converter.unstructure(result)
        
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        print("\n🎉 RAG2 simulation completed successfully!")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main())
