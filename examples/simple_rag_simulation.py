#!/usr/bin/env python3
"""
Simple end-to-end RAG-based conversation simulation example with in-memory vector store.

This is a simplified version that uses InMemoryVectorStore for vector storage with zero external dependencies.
It demonstrates the same workflow as a full example but with minimal setup requirements.

When running multiple simulations, you can instead split the vector store initialization into a separate script
to avoid re-initializing the vector store each time, which can be time-consuming.

Note on logging vs. print:
- logger.info/debug/error: Used for internal process information and debugging
- print(): Used for pretty user-facing output and summaries

Usage:
    python examples/simple_rag_simulation.py
    
Requirements:
    - OpenAI API key set in environment (OPENAI_API_KEY)
    - No additional vector store dependencies required!
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from conversation_simulator.models import (
    Conversation,
    Outcomes,
    ParticipantRole,
)
from conversation_simulator.models.results import SimulationSessionResult
from conversation_simulator.participants.agent.rag import RagAgentFactory
from conversation_simulator.participants.customer.rag import RagCustomerFactory
from conversation_simulator.rag import conversations_to_langchain_documents
from conversation_simulator.simulation.session_builder import SimulationSessionBuilder
from conversation_simulator.util.structure import converter

# Get module logger
logger = logging.getLogger(__name__)

def load_sample_conversations() -> list[Conversation]:
    """Load sample conversations from the test data file using cattrs.
    
    Returns:
        List of sample Conversation objects
    """
    # Path to the sample data file
    data_file = Path(__file__).parent.parent / "tests" / "data" / "dch2_sampled_dataset.json"
    
    logger.info(f"Loading sample conversations from {data_file}")
    
    if not data_file.exists():
        raise FileNotFoundError(f"Sample data file not found: {data_file}")
    
    try:
        with data_file.open('r', encoding='utf-8') as f:
            data = json.load(f)
        # Convert JSON data to Conversation objects using cattrs
        conversations: list[Conversation] = converter.structure(data['conversations'], list[Conversation])
        logger.info(f"Loaded {len(conversations)} sample conversations")
        return conversations
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ValueError(f"Failed to parse conversation data: {e}") from e


def extract_outcomes_from_conversations(conversations: list[Conversation]) -> Outcomes:
    """Extract unique outcomes from conversations.
    
    Args:
        conversations: List of conversations with outcomes
        
    Returns:
        Outcomes object containing all unique outcomes found
    """
    unique_outcomes = {}
    
    for conversation in conversations:
        if conversation.outcome:
            outcome_name = conversation.outcome.name
            if outcome_name not in unique_outcomes:
                unique_outcomes[outcome_name] = conversation.outcome
    
    if not unique_outcomes:
        raise ValueError("No outcomes found in conversations. Cannot proceed with simulation.")
    
    outcomes_tuple = tuple(unique_outcomes.values())
    logger.info(f"Extracted {len(outcomes_tuple)} unique outcomes: {[o.name for o in outcomes_tuple]}")
    return Outcomes(outcomes=outcomes_tuple)


async def build_simple_vector_stores(
    reference_conversations: list[Conversation],
    embeddings_model: OpenAIEmbeddings
) -> tuple[InMemoryVectorStore, InMemoryVectorStore]:
    """Build simple in-memory vector stores for agent and customer messages.
    
    Args:
        reference_conversations: List of reference conversations
        embeddings_model: OpenAI embeddings model instance
        
    Returns:
        Tuple of (agent_vector_store, customer_vector_store)
    """
    logger.info("Building in-memory vector stores from reference conversations")
    
    # Convert conversations to documents for each role
    agent_documents = conversations_to_langchain_documents(
        reference_conversations, 
        role=ParticipantRole.AGENT
    )
    customer_documents = conversations_to_langchain_documents(
        reference_conversations, 
        role=ParticipantRole.CUSTOMER
    )
    
    logger.info(f"Created {len(agent_documents)} agent documents and {len(customer_documents)} customer documents")
    
    # Create in-memory vector stores
    agent_vector_store = InMemoryVectorStore.from_documents(
        documents=agent_documents,
        embedding=embeddings_model
    )
    
    customer_vector_store = InMemoryVectorStore.from_documents(
        documents=customer_documents,
        embedding=embeddings_model
    )
    
    logger.info("In-memory vector stores created successfully")
    return agent_vector_store, customer_vector_store


async def run_rag_simulation(
    embeddings_model_name: str,
    chat_model_name: str,
    outcomes: Outcomes,
    reference_conversations: list[Conversation],
    number_of_simulations: int = 20,
) -> SimulationSessionResult:
    """Run RAG-based simulation with specified parameters.
    
    This is the core function that implements end-to-end RAG simulation.
    
    Args:
        embeddings_model_name: Name of the OpenAI embeddings model to use
        chat_model_name: Name of the OpenAI chat model to use
        outcomes: Possible conversation outcomes
        reference_conversations: List of reference conversations for RAG
        number_of_simulations: Number of conversations to simulate (default: 20)

    Returns:
        SimulationSessionResult containing the simulation outcomes
    """
    logger.info(f"Starting RAG simulation with model: {chat_model_name}")
    
    # Initialize OpenAI components
    chat_model = ChatOpenAI(model=chat_model_name, temperature=0.0)
    embeddings_model = OpenAIEmbeddings(model=embeddings_model_name)
    
    # Build vector stores
    agent_vector_store, customer_vector_store = await build_simple_vector_stores(
        reference_conversations, 
        embeddings_model
    )
    
    # Create participant factories
    agent_factory = RagAgentFactory(
        model=chat_model,
        agent_vector_store=agent_vector_store
    )
    
    customer_factory = RagCustomerFactory(
        model=chat_model,
        customer_vector_store=customer_vector_store
    )
    
    # Build simulation session
    session = SimulationSessionBuilder(
        chat_model=chat_model,
        agent_factory=agent_factory,
        customer_factory=customer_factory,
        outcomes=outcomes,
        session_name="RAG Simulation",
        session_description=f"RAG-based simulation using {chat_model_name} with {len(reference_conversations)} reference conversations",
        max_messages=20,  # Reasonable limit for the example
    ).build()
    
    logger.info(f"Running simulation with max {session.max_messages} messages per conversation")
    
    # Run simulation with reference conversations
    # Sample number_of_simulations conversations for simulation
    if len(reference_conversations) > number_of_simulations:
        base_conversations = reference_conversations[:number_of_simulations]
        logger.info(f"Using only the first {number_of_simulations} reference conversations for simulation")
    else:
        base_conversations = reference_conversations
    result = await session.run_simulation(base_conversations)
    
    logger.info(f"Simulation completed. Generated {len(result.simulated_conversations)} conversations")
    return result


def print_simple_summary(result: SimulationSessionResult) -> None:
    """Print a simple summary of the simulation results.
    
    Args:
        result: SimulationSessionResult to summarize
    """
    print("\n" + "="*40)
    print("SIMULATION RESULTS")
    print("="*40)
    
    print(f"Session name: {result.session_name}")
    print(f"Original conversations: {len(result.original_conversations)}")
    print(f"Simulated conversations: {len(result.simulated_conversations)}")
    
    if result.simulated_conversations:
        # Count outcomes
        outcome_counts: dict[str, int] = {}
        total_messages = 0
        
        for sim_conv in result.simulated_conversations:
            # Count outcome
            outcome_name = sim_conv.conversation.outcome.name if sim_conv.conversation.outcome else "unknown"
            outcome_counts[outcome_name] = outcome_counts.get(outcome_name, 0) + 1
            
            # Count messages
            total_messages += len(sim_conv.conversation.messages)
        
        avg_messages = total_messages / len(result.simulated_conversations)
        print(f"Average messages per conversation: {avg_messages:.1f}")
        
        print("\nOutcome distribution:")
        for outcome_name, count in sorted(outcome_counts.items()):
            percentage = (count / len(result.simulated_conversations)) * 100
            print(f"  {outcome_name}: {count} ({percentage:.1f}%)")
        
        # Show a sample conversation
        if result.simulated_conversations:
            sample_conv = result.simulated_conversations[0].conversation
            print(f"\nSample conversation ({len(sample_conv.messages)} messages):")
            for i, msg in enumerate(sample_conv.messages[:4]):  # Show first 4 messages
                content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                print(f"  {i+1}. {msg.sender.value}: {content_preview}")
            if len(sample_conv.messages) > 4:
                print(f"  ... and {len(sample_conv.messages) - 4} more messages")
    
    print("="*40)


async def main() -> None:
    """Main function to run the simple RAG simulation example."""
    # Validate OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Load sample data
        reference_conversations = load_sample_conversations()
        
        # Extract outcomes from the conversations
        outcomes = extract_outcomes_from_conversations(reference_conversations)
        
        # Run the simulation with the core function
        result = await run_rag_simulation(
            embeddings_model_name="text-embedding-3-small",
            chat_model_name="gpt-4o-mini",
            outcomes=outcomes,
            reference_conversations=reference_conversations,
            number_of_simulations=20,  # Limit to 20 for this example
        )
        
        # Display results (using print for pretty user output)
        print_simple_summary(result)
        
        # Optionally save results
        output_file = Path("simple_simulation_results.json")
        result_dict = converter.unstructure(result)
        
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}\n")
        logger.info("Simple RAG simulation completed successfully!")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
