"""
🤖 Conversation Simulator Runner

A Streamlit page for running RAG-based conversation simulations.
"""

import streamlit as st
import asyncio
from typing import Any

# Import conversation simulator components
from agentune.simulate.models import Conversation, Outcomes
from agentune.simulate.models.results import SimulationSessionResult, ConversationResult
from agentune.simulate.models.outcome import Outcome
from agentune.simulate.participants.agent.rag import RagAgentFactory
from agentune.simulate.participants.customer.rag import RagCustomerFactory
from agentune.simulate.simulation.progress import ProgressCallback
from agentune.simulate.rag import conversations_to_langchain_documents
from agentune.simulate.intent_extraction.zeroshot import ZeroshotIntentExtractor
from agentune.simulate.outcome_detection.rag.rag import RAGOutcomeDetector
from agentune.simulate.simulation.adversarial.zeroshot import ZeroShotAdversarialTester

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from agentune.simulate.simulation.simulation_session import SimulationSession

# Import helper functions
from helper import (
    load_conversation_data, conversations_to_dataframe, select_from_dataframe,
    display_conversation, get_openai_models, format_results_for_download,
    validate_api_key, extract_unique_outcomes, get_llm_callbacks
)


class StreamlitProgressCallback(ProgressCallback):
    """Progress callback that updates Streamlit UI elements in real-time."""
    
    def __init__(self, progress_placeholder, status_placeholder):
        """Initialize with Streamlit placeholders for updating UI.
        
        Args:
            progress_placeholder: Streamlit placeholder for progress bar
            status_placeholder: Streamlit placeholder for status text
        """
        self.progress_placeholder = progress_placeholder
        self.status_placeholder = status_placeholder
        self.total_scenarios = 0
        self.completed_scenarios = 0
        self.failed_scenarios = 0
        self.current_phase = "Initializing..."
        
    def on_generated_scenarios(self, scenarios) -> None:
        """Called when scenarios are generated."""
        self.total_scenarios = len(scenarios)
        self.current_phase = "Generated scenarios"
        self._update_display()
    
    def on_scenario_start(self, scenario) -> None:
        """Called when a scenario starts."""
        self.current_phase = f"Running scenario: {scenario.id}"
        self._update_display()
    
    def on_scenario_complete(self, scenario, result: ConversationResult) -> None:
        """Called when a scenario completes successfully."""
        self.completed_scenarios += 1
        self.current_phase = f"Completed {self.completed_scenarios}/{self.total_scenarios} scenarios {f'({self.failed_scenarios} failed)' if self.failed_scenarios > 0 else ''}"
        self._update_display()
    
    def on_scenario_failed(self, scenario, exception: Exception) -> None:
        """Called when a scenario fails."""
        self.failed_scenarios += 1
        self.completed_scenarios += 1
        self.current_phase = f"Completed {self.completed_scenarios}/{self.total_scenarios} scenarios ({self.failed_scenarios} failed)"
        self._update_display()
    
    def on_all_scenarios_complete(self) -> None:
        """Called when all scenarios are complete."""
        self.current_phase = f"All scenarios complete. ({self.completed_scenarios} completed, {self.failed_scenarios} failed) Analyzing results..."
        self._update_display()
    
    def _update_display(self):
        """Update the Streamlit UI elements."""
        try:
            if self.total_scenarios > 0:
                progress = self.completed_scenarios / self.total_scenarios
                self.progress_placeholder.progress(progress, text=f"Progress: {self.completed_scenarios}/{self.total_scenarios}")
            else:
                self.progress_placeholder.progress(0, text="Preparing...")
            
            self.status_placeholder.text(f"Status: {self.current_phase}")
        except Exception:
            # Silently ignore any UI update errors to avoid breaking simulation
            pass


def select_model_with_temperature(
    label: str,
    chat_models: list[str],
    default_model: str,
    reasoning_models: list[str],
    help_text: str,
    key_prefix: str = ""
) -> dict[str, Any]:
    """Select a model and optionally its temperature, returning model kwargs.
    
    Args:
        label: Label for the model selection
        chat_models: List of available chat models
        default_model: Default model to select
        reasoning_models: List of reasoning models (no temperature)
        help_text: Help text for the model selection
        key_prefix: Optional prefix for widget keys to ensure uniqueness
        
    Returns:
        Dict containing model kwargs ready for ChatOpenAI
    """
    col1, col2 = st.columns([3, 1])
    model = col1.selectbox(
        label,
        chat_models,
        index=chat_models.index(default_model),
        help=help_text,
        key=f"{key_prefix}_model" if key_prefix else None
    )

    model_kwargs: dict[str, Any] = {'model': model}

    if model not in reasoning_models:
        temp = col2.number_input(
            "temp",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help=f"Temperature for {label.lower()}",
            key=f"{key_prefix}_temp" if key_prefix else None
        )
        model_kwargs['temperature'] = temp
    
    return model_kwargs


def initialize_sidebar():
    """Initialize sidebar with configuration options."""
    st.sidebar.header("⚙️ Configuration")
    
    # File upload
    st.sidebar.subheader("📁 Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload conversation data file",
        type=['json'],
        help="Upload JSON file containing conversation data"
    )
    
    # Model selection
    st.sidebar.subheader("🤖 Models")
    models = get_openai_models()
    
    # Chat models for selection (exclude embedding models)
    chat_models = []
    for category, model_list in models.items():
        if "Embedding" not in category:
            chat_models.extend(model_list)

    default_model: str = "gpt-4o-2024-08-06"

    if default_model not in chat_models:
        raise Exception(f"Default model '{default_model}' is not on the list of available chat models.")
    
    # Default adversarial model (reasoning model)
    reasoning_models = models.get("Reasoning Models", [])
    default_adversarial: str = reasoning_models[-1] if reasoning_models else default_model

    default_embedding: str = "text-embedding-3-small"
    
    # Settings mode toggle
    use_advanced = st.sidebar.toggle(
        "🔧 Advanced Settings",
        value=False,
        help="Enable advanced configuration with individual component models"
    )
    
    # Advanced settings - individual models for each component
    st.sidebar.markdown("**Models:**")
    if use_advanced:
        
        with st.sidebar:
            agent_model_kwargs = select_model_with_temperature(
                "Agent Model",
                chat_models,
                default_model,
                reasoning_models,
                "Model for the agent participant",
                "agent"
            )

            customer_model_kwargs = select_model_with_temperature(
                "Customer Model",
                chat_models,
                default_model,
                reasoning_models,
                "Model for the customer participant",
                "customer"
            )
            
            intent_model_kwargs = select_model_with_temperature(
                "Intent Classification",
                chat_models,
                default_model,
                reasoning_models,
                "Model for intent extraction",
                "intent"
            )

            outcome_model_kwargs = select_model_with_temperature(
                "Outcome Detection",
                chat_models,
                default_model,
                reasoning_models,
                "Model for outcome detection",
                "outcome"
            )
            
            adversarial_model_kwargs = select_model_with_temperature(
                "Adversarial Model",
                chat_models,
                default_adversarial,
                reasoning_models,
                "Model for adversarial testing",
                "adversarial"
            )
    else:
        # Basic settings - simplified interface
        with st.sidebar:
            basic_model_kwargs = select_model_with_temperature(
                "Main Model",
                chat_models,
                default_model,
                reasoning_models,
                "Model for agent, customer, intent, and outcome detection",
                "basic"
            )

            adversarial_model_kwargs = select_model_with_temperature(
                "Adversarial Model",
                chat_models,
                default_adversarial,
                reasoning_models,
                "Model for adversarial testing (recommended: reasoning model)",
                "adversarial_basic"
            )
        
        # Use basic model for all components
        agent_model_kwargs = basic_model_kwargs.copy()
        customer_model_kwargs = basic_model_kwargs.copy()
        intent_model_kwargs = basic_model_kwargs.copy()
        outcome_model_kwargs = basic_model_kwargs.copy()
        
    # Embedding model (always shown)
    embedding_models = models.get("Embedding Models", [default_embedding])
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        embedding_models,
        index=embedding_models.index(default_embedding),
        help="Model for generating embeddings for RAG",
        key="embedding_model"
    )
    
    # Simulation parameters
    st.sidebar.markdown("**Simulation Parameters:**")
    
    max_messages = st.sidebar.number_input(
        "Max Messages per Conversation",
        min_value=5,
        max_value=100,
        value=20,
        help="Maximum number of messages in each simulated conversation"
    )
    
    max_concurrent_conversations = st.sidebar.number_input(
        "Max Concurrent Conversations",
        min_value=1,
        max_value=200,
        value=50,
        help="Maximum number of conversations to run in parallel"
    )

    session_name = st.sidebar.text_input(
        "Session Name",
        value="RAG Simulation",
        help="Name for this simulation session"
    )
    
    return {
        'uploaded_file': uploaded_file,
        'embedding_model': embedding_model,
        'max_messages': max_messages,
        'max_concurrent_conversations': max_concurrent_conversations,
        'session_name': session_name,
        'agent_model_kwargs': agent_model_kwargs,
        'customer_model_kwargs': customer_model_kwargs,
        'intent_model_kwargs': intent_model_kwargs,
        'outcome_model_kwargs': outcome_model_kwargs,
        'adversarial_model_kwargs': adversarial_model_kwargs
    }


async def build_vector_store(
    reference_conversations: list[Conversation],
    embeddings_model: OpenAIEmbeddings
) -> InMemoryVectorStore:
    """Build a single vector store from reference conversations."""
    
    # Convert conversations to documents (without role filtering for shared vector store)
    documents = conversations_to_langchain_documents(reference_conversations)
    
    # Create a single in-memory vector store for all components
    vector_store = InMemoryVectorStore.from_documents(
        documents=documents,
        embedding=embeddings_model
    )
    
    return vector_store


async def run_simulation(
    selected_conversations: list[Conversation],
    all_conversations: list[Conversation],
    config: dict[str, Any],
    progress_callback: ProgressCallback
) -> SimulationSessionResult:
    """Run the conversation simulation."""
    
    # Get logging callback tracer
    callbacks = get_llm_callbacks(config['session_name'])
    
    # Initialize models with logging callbacks
    agent_model = ChatOpenAI(**config['agent_model_kwargs'], callbacks=callbacks)
    customer_model = ChatOpenAI(**config['customer_model_kwargs'], callbacks=callbacks)
    intent_model = ChatOpenAI(**config['intent_model_kwargs'], callbacks=callbacks)
    outcome_model = ChatOpenAI(**config['outcome_model_kwargs'], callbacks=callbacks)
    adversarial_model = ChatOpenAI(**config['adversarial_model_kwargs'], callbacks=callbacks)
    embeddings_model = OpenAIEmbeddings(model=config['embedding_model'])
    
    # Build single vector store (using all conversations for context)
    vector_store = await build_vector_store(
        all_conversations,
        embeddings_model
    )
    
    # Create participant factories using the single shared vector store
    agent_factory = RagAgentFactory(
        model=agent_model,
        agent_vector_store=vector_store
    )
    
    customer_factory = RagCustomerFactory(
        model=customer_model,
        customer_vector_store=vector_store
    )

    intent_extractor = ZeroshotIntentExtractor(intent_model, max_concurrency=config['max_concurrent_conversations'])
    outcome_detector = RAGOutcomeDetector(
        model=outcome_model,
        vector_store=vector_store
    )
    adversarial_tester = ZeroShotAdversarialTester(adversarial_model, max_concurrency=config['max_concurrent_conversations'])
    
    # Extract outcomes
    unique_outcome_dicts = extract_unique_outcomes(selected_conversations)
    if not unique_outcome_dicts:
        raise ValueError("No outcomes found in selected conversations")
    
    # Convert outcome dictionaries to Outcome objects
    unique_outcomes = []
    for outcome_dict in unique_outcome_dicts:
        outcome_obj = Outcome(
            name=outcome_dict.get('name', 'unknown'),
            description=outcome_dict.get('description', '')
        )
        unique_outcomes.append(outcome_obj)
    
    outcomes = Outcomes(outcomes=tuple(unique_outcomes))
    
    # Build simulation session with custom models
    session_description = (
        f"RAG simulation using {config['agent_model_kwargs']['model']} (agent), "
        f"{config['customer_model_kwargs']['model']} (customer), {config['intent_model_kwargs']['model']} (intent), "
        f"{config['outcome_model_kwargs']['model']} (outcome), {config['adversarial_model_kwargs']['model']} (adversarial)"
    )
    
    session = SimulationSession(
        outcomes=outcomes,
        agent_factory=agent_factory,
        customer_factory=customer_factory,
        intent_extractor=intent_extractor,
        outcome_detector=outcome_detector,
        adversarial_tester=adversarial_tester,
        session_name=config['session_name'],
        session_description=session_description,
        max_messages=config['max_messages'],
        max_concurrent_conversations=config['max_concurrent_conversations'],
        return_exceptions=True,
        progress_callback=progress_callback
    )
    
    # Run simulation
    result = await session.run_simulation(selected_conversations)
    
    return result


def main():
    """Main function for the simulation runner page."""
    
    st.title("🤖 Conversation Simulator Runner")
    st.markdown("Run RAG-based conversation simulations with custom settings")
    
    # Validate API key
    if not validate_api_key():
        return
    
    # Initialize sidebar
    config = initialize_sidebar()
    
    if config['uploaded_file'] is None:
        st.markdown("""
        ## Welcome to the Conversation Simulator Runner! 🚀
        
        To get started:
        1. **Upload conversation data** using the file uploader in the sidebar
        2. **Configure models** and settings for your simulation
        3. **Select conversations** to use as scenarios for simulation
        4. **Run the simulation** and download results
        
        ### Features:
        - 🤖 **Flexible Model Selection**: Use different models for agent and customer
        - 🎯 **Custom Scenarios**: Select specific conversations to simulate
        - ⚙️ **Advanced Settings**: Control temperature, message limits, and more
        - 📊 **Real-time Progress**: Monitor simulation progress
        - 💾 **Download Results**: Get results in JSON format for analysis
        
        ### Requirements:
        - Valid OpenAI API key in environment variables
        - Conversation data in JSON format
        """)
        return
    
    # Load conversation data
    with st.spinner("Loading conversation data..."):
        conversations = load_conversation_data(config['uploaded_file'])
        if conversations is None:
            return
    
    st.success(f"✅ Loaded {len(conversations)} conversations")
    
    # Convert to DataFrame for selection
    df = conversations_to_dataframe(conversations)
    
    # Show conversation selection
    st.header("📋 Select Conversations to Simulate")
    st.markdown("Choose the conversations you want to use as scenarios for simulation:")
    
    selected_conversations, selected_indices = select_from_dataframe(
        df, "Conversations for Simulation", multi_rows=True, random_select=True
    )
    
    if not selected_conversations:
        st.info("Please select one or more conversations to simulate.")
        return
    
    st.success(f"Selected {len(selected_conversations)} conversations for simulation")
    
    # Show preview of selected conversations
    with st.expander("📖 Preview Selected Conversations", expanded=False):
        for i, conv in enumerate(selected_conversations[:3]):  # Show first 3
            st.markdown(f"**Conversation {i + 1}:**")
            display_conversation(conv['conversation_data'], f"Preview {i + 1}")
            if i < len(selected_conversations) - 1:
                st.markdown("---")
        
        if len(selected_conversations) > 3:
            st.info(f"... and {len(selected_conversations) - 3} more conversations")
    
    # Run simulation
    st.header("🚀 Run Simulation")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button(
            "▶️ Start Simulation",
            type="primary",
            use_container_width=True,
            help="Start the conversation simulation with selected settings",
            disabled=not bool(selected_conversations)
        )
    
    # Handle simulation execution
    if run_button:
        # Extract conversation data for simulation
        simulation_conversations = [conv['conversation_data'] for conv in selected_conversations]
        
        # Create progress display elements
        progress_container = st.container()
        with progress_container:
            st.markdown("### 🔄 Simulation Progress")
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
        # Create progress callback
        progress_callback = StreamlitProgressCallback(progress_placeholder, status_placeholder)
        
        try:
            # Run the simulation with progress callback
            result = asyncio.run(run_simulation(simulation_conversations, conversations, config, progress_callback))
            
            # Clear progress display
            progress_container.empty()
            
            # Store results in session state
            st.session_state['simulation_result'] = result
            st.session_state['simulation_config'] = config
            st.session_state['simulation_conversations'] = simulation_conversations
            
            st.success("🎉 Simulation completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Simulation failed: {str(e)}")
            st.markdown("Please check your configuration and try again.")
            with st.expander("Error Details"):
                st.code(str(e))
            # Clear any existing results on failure
            if 'simulation_result' in st.session_state:
                del st.session_state['simulation_result']
            raise e
    
    # Display results if they exist in session state
    if 'simulation_result' in st.session_state:
        result = st.session_state['simulation_result']
        config = st.session_state['simulation_config']
        simulation_conversations = st.session_state['simulation_conversations']
        
        # Show results summary
        st.header("📊 Simulation Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Input Conversations", len(simulation_conversations))
        
        with col2:
            st.metric("Generated Conversations", len(result.simulated_conversations))
            
        with col3:
            # Use analysis_result for more accurate average
            avg_messages = result.analysis_result.message_distribution_comparison.simulated_stats.mean_messages
            st.metric("Avg Messages", f"{avg_messages:.1f}")
        
        with col4:
            # Calculate duration
            if result.started_at and result.completed_at:
                duration = result.completed_at - result.started_at
                st.metric("Duration", f"{duration.total_seconds():.1f}s")
            else:
                st.metric("Duration", "Unknown")
        
        # Show outcome distribution using analysis_result
        outcome_comparison = result.analysis_result.outcome_comparison
        
        if outcome_comparison:
            st.subheader("🎯 Outcome Distribution Comparison")
            
            # Show simulated distribution
            simulated_dist = outcome_comparison.simulated_distribution
            if simulated_dist and simulated_dist.outcome_counts:
                st.markdown("**Simulated Conversations:**")
                total_simulated = simulated_dist.total_conversations
                for outcome, count in sorted(simulated_dist.outcome_counts.items()):
                    percentage = (count / total_simulated * 100) if total_simulated > 0 else 0
                    st.write(f"**{outcome}**: {count} conversations ({percentage:.1f}%)")
        
        # Show sample conversation
        if result.simulated_conversations:
            st.subheader("💬 Sample Generated Conversation")
            sample_conv = result.simulated_conversations[0].conversation
            
            with st.expander("View Sample Conversation", expanded=False):
                # Display the conversation directly since display_conversation now handles Conversation objects
                display_conversation(sample_conv, "Sample Generated Conversation")
        
        # Download results
        st.header("💾 Download Results")
        
        # Format results for download
        json_str, filename = format_results_for_download(
            result,
            config['session_name'].replace(' ', '_').lower()
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="📥 Download Simulation Results",
                data=json_str,
                file_name=filename,
                mime="application/json",
                type="primary",
                use_container_width=True
            )
        
        st.info("💡 You can analyze these results using the **💬 Conversation Simulator Results Analyzer** page.")
        
        # Option to clear results
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🗑️ Clear Results", help="Clear the current simulation results"):
                del st.session_state['simulation_result']
                del st.session_state['simulation_config']
                del st.session_state['simulation_conversations']
                st.rerun()


if __name__ == "__main__":
    main()
