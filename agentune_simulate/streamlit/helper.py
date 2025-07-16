"""
Helper functions for the Conversation Simulator Streamlit apps.

Common utilities shared across different pages of the application.
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import random
import os
import re
from datetime import datetime
from typing import Any

from agentune.simulate.models.conversation import Conversation
from agentune.simulate.models.results import SimulationSessionResult
from agentune.simulate.util.structure import converter


def get_llm_callbacks() -> list:
    """
    Configure and return callback handlers for LLM logging and tracking.

    Returns:
        list: List containing LangChain logging callback and Opik tracer
    """
    return []


def load_simulation_results(uploaded_file) -> SimulationSessionResult | None:
    """Load simulation results from uploaded JSON file."""
    try:
        content = uploaded_file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        json_content = json.loads(content)
        result: SimulationSessionResult = converter.structure(json_content, SimulationSessionResult)
        return result
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON file: {e}")
        return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def load_conversation_data(uploaded_file) -> list[Conversation]:
    """Load conversation data from uploaded JSON file.
    
    Returns:
        List of sample Conversation objects
    """
    try:
        content = uploaded_file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        data = json.loads(content)
        # Convert JSON data to Conversation objects using cattrs
        conversations: list[Conversation] = converter.structure(data['conversations'], list[Conversation])
        return conversations
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        st.error(f"Failed to parse conversation data: {e}")
        return []


def extract_conversation_data(results: SimulationSessionResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract conversation data into DataFrames for analysis."""
    
    # Extract original conversations
    original_data = []
    for orig_conv in results.original_conversations:
        conversation = orig_conv.conversation
        
        # Handle outcome safely
        outcome_name = conversation.outcome.name if conversation.outcome else 'unknown'
        
        original_data.append({
            'id': orig_conv.id,
            'type': 'Original',
            'num_messages': len(conversation.messages),
            'outcome': outcome_name,
            'first_message': conversation.messages[0].content[:100] + "..." if conversation.messages else "",
            'conversation_data': conversation
        })
    
    # Extract simulated conversations
    simulated_data = []
    for sim_conv in results.simulated_conversations:
        conversation = sim_conv.conversation
        
        # Handle outcome safely
        outcome_name = conversation.outcome.name if conversation.outcome else 'unknown'
        
        simulated_data.append({
            'id': sim_conv.id,
            'type': 'Simulated',
            'scenario_id': sim_conv.scenario_id,
            'original_id': sim_conv.original_conversation_id,
            'num_messages': len(conversation.messages),
            'outcome': outcome_name,
            'first_message': conversation.messages[0].content[:100] + "..." if conversation.messages else "",
            'conversation_data': conversation
        })
    
    original_df = pd.DataFrame(original_data)
    simulated_df = pd.DataFrame(simulated_data)
    
    return original_df, simulated_df


def conversations_to_dataframe(conversations: list[Conversation]) -> pd.DataFrame:
    """Convert conversation data to DataFrame for display and selection."""
    data = []
    for i, conv in enumerate(conversations):
        # Handle outcome safely
        outcome_name = conv.outcome.name if conv.outcome else 'unknown'
        
        data.append({
            'index': i,
            'id': f'conversation_{i}',  # Generate ID since Conversation objects don't have an id field
            'num_messages': len(conv.messages),
            'outcome': outcome_name,
            'first_message': conv.messages[0].content[:100] + "..." if conv.messages else "",
            'conversation_data': conv
        })
    
    return pd.DataFrame(data)


def show_conversation_filters(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Show filter controls and return filtered dataframe."""
    st.subheader(f"🔍 Filter {table_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Outcome filter
        outcomes = ['All'] + sorted(df['outcome'].unique().tolist())
        selected_outcome = st.selectbox("Filter by outcome", outcomes, key=f"outcome_{table_name}")
        
    with col2:
        # Message count filter
        min_messages, max_messages = int(df['num_messages'].min()), int(df['num_messages'].max())
        
        # Handle case where all conversations have the same number of messages
        if min_messages == max_messages:
            st.write(f"All conversations have {min_messages} messages")
            message_range = (min_messages, max_messages)
        else:
            message_range = st.slider(
                "Filter by message count",
                min_messages,
                max_messages,
                (min_messages, max_messages),
                key=f"messages_{table_name}"
            )
    
    # Text search filter row
    search_col1, search_col2, search_col3 = st.columns([3, 1, 1])
    
    with search_col1:
        search_text = st.text_input(
            "🔍 Search in conversation text",
            placeholder="Enter text to search in conversation messages...",
            key=f"search_text_{table_name}",
            help="Search for specific words or phrases within the conversation messages"
        )
    
    with search_col2:
        case_sensitive = st.checkbox(
            "Case sensitive",
            value=False,
            key=f"case_sensitive_{table_name}",
            help="Enable case-sensitive search"
        )
    
    with search_col3:
        whole_words = st.checkbox(
            "Whole words",
            value=False,
            key=f"whole_words_{table_name}",
            help="Match whole words only"
        )
    
    # Check if filters changed and clear random selection if so
    filter_state_key = f"filter_state_{table_name}"
    current_filter_state = (selected_outcome, message_range, search_text, case_sensitive, whole_words)
    
    if filter_state_key in st.session_state:
        if st.session_state[filter_state_key] != current_filter_state:
            # Filters changed, clear random selection
            random_selection_key = f"random_selection_indices_{table_name}"
            if random_selection_key in st.session_state:
                del st.session_state[random_selection_key]
    
    st.session_state[filter_state_key] = current_filter_state
    
    # Apply filters
    filtered_df = df.copy()
    if selected_outcome != 'All':
        filtered_df = filtered_df[filtered_df['outcome'] == selected_outcome]
    
    filtered_df = filtered_df[
        (filtered_df['num_messages'] >= message_range[0]) & (filtered_df['num_messages'] <= message_range[1])
    ]
    
    # Apply text search filter
    if search_text.strip():
        def search_in_conversation(row):
            """Search for text within conversation messages."""
            try:
                conversation_data = row['conversation_data']
                # Combine all message content into one text
                full_text = ' '.join([msg.content for msg in conversation_data.messages])
                
                # Prepare search text
                search_term = search_text.strip()
                text_to_search = full_text
                
                # Apply case sensitivity
                if not case_sensitive:
                    search_term = search_term.lower()
                    text_to_search = text_to_search.lower()
                
                # Apply whole words matching
                if whole_words:
                    # Use word boundaries for whole word matching
                    pattern = r'\b' + re.escape(search_term) + r'\b'
                    return bool(re.search(pattern, text_to_search))
                else:
                    return search_term in text_to_search
                    
            except Exception:
                # If there's any error in processing, exclude the row
                return False
        
        # Apply the search filter
        search_mask = filtered_df.apply(search_in_conversation, axis=1)
        filtered_df = filtered_df[search_mask]
    
    # Reset index to ensure continuous 0-based indexing
    filtered_df = filtered_df.reset_index(drop=True)
    
    return filtered_df


def show_random_selection_controls(length: int, table_name: str) -> list[int] | None:
    """Show random selection controls and return selected indices for the given length."""
    
    st.subheader("🎲 Random Selection")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_conversations = st.number_input(
            "Number of conversations to select",
            min_value=1,
            max_value=length,
            value=min(10, length),
            help="How many conversations to randomly select from filtered results"
        )
    
    with col2:
        random_seed = st.number_input(
            "Random seed",
            min_value=0,
            value=42,
            help="Seed for reproducible random selection"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        random_select_button = st.button(
            "🎲 Random Select",
            help="Randomly select the specified number of conversations",
            use_container_width=True,
            key=f"random_select_{table_name}"
        )
    
    # Handle random selection
    session_key = f"random_selection_indices_{table_name}"
    
    if random_select_button:
        random.seed(random_seed)
        if length <= num_conversations:
            # Select all if requested number is >= total filtered
            selected_indices = list(range(length))
        else:
            # Random selection from range
            selected_indices = random.sample(range(length), num_conversations)
        
        st.session_state[session_key] = selected_indices
        st.success(f"🎲 Randomly selected {len(selected_indices)} conversations from {length} filtered conversations")
    
    return st.session_state.get(session_key, None)


def select_from_dataframe(
    df: pd.DataFrame,
    table_name: str,
    multi_rows: bool = False,
    random_select: bool = False
) -> tuple[Any, Any]:
    """Select conversations from dataframe with filtering."""
    
    if df.empty:
        st.warning(f"No {table_name} available.")
        return ([] if multi_rows else None), ([] if multi_rows else None)
    
    # Show filters
    filtered_df = show_conversation_filters(df, table_name)
    
    if filtered_df.empty:
        st.warning("No conversations match the selected filters.")
        return ([] if multi_rows else None), ([] if multi_rows else None)
    
    # Handle multi-row selection differently
    if multi_rows:
        # Show random selection controls if enabled
        default_selection = None
        if random_select:
            default_selection = show_random_selection_controls(len(filtered_df), table_name)
        
        # Display selection table with data_editor
        st.subheader(f"📋 Select {table_name}")
        
        # Show summary columns for selection
        display_df = filtered_df[['id', 'outcome', 'num_messages', 'first_message']].copy()
        
        # Add a 'selected' boolean column at the start
        display_df.insert(0, 'selected', False)
        
        # Handle default selection
        if default_selection is not None:
            for idx in default_selection:
                if idx < len(display_df):
                    selected_col_idx = display_df.columns.get_loc('selected')
                    if isinstance(selected_col_idx, int):
                        display_df.iloc[idx, selected_col_idx] = True
        
        # Use data_editor for selection
        edited_df = st.data_editor(
            data=display_df,
            use_container_width=True,
            key=f"select_{table_name}",
            hide_index=True,
            column_config={
                "selected": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select conversations to use",
                    default=False,
                    width="small"
                ),
                "id": st.column_config.TextColumn("ID", width="medium"),
                "outcome": st.column_config.TextColumn("Outcome", width="medium"),
                "num_messages": st.column_config.NumberColumn("Messages", width="small"),
                "first_message": st.column_config.TextColumn("First Message", width="large")
            },
            disabled=["id", "outcome", "num_messages", "first_message"]  # Only allow editing the 'selected' column
        )
        
        # Get selected rows
        selected_rows = edited_df[edited_df['selected']].index.tolist()
        
        if selected_rows:
            # Multi selection
            out_list, id_list = [], []
            for idx in selected_rows:
                row_series = filtered_df.loc[idx]
                row_dict = row_series.to_dict()
                out_list.append(row_dict)
                # Use the positional index as the ID since we reset index
                id_list.append(idx)
            return out_list, id_list
        else:
            st.info("Select one or more rows using the checkboxes.")
            return [], []
    
    else:
        # Single selection - use dataframe selection
        st.subheader(f"📋 Select {table_name}")
        
        # Show summary columns for selection
        display_df = filtered_df[['id', 'outcome', 'num_messages', 'first_message']].copy()
        
        selection_result = st.dataframe(
            data=display_df,
            use_container_width=True,
            key=f"select_{table_name}",
            on_select="rerun",
            selection_mode="single-row"
        )
        
        selected_rows = selection_result.selection['rows']  # type: ignore[attr-defined]
        
        if selected_rows:
            idx = selected_rows[0]
            row_series = filtered_df.loc[idx]
            row_dict = row_series.to_dict()
            return row_dict, idx

        # Nothing selected
        st.info("Select a row to view conversation details.")
        return None, None


def display_conversation(conversation: Conversation, title: str = "Conversation"):
    """Display a conversation in a chat-like format."""
    
    st.subheader(f"💬 {title}")
    
    # Conversation metadata
    outcome_name = conversation.outcome.name if conversation.outcome else 'unknown'
    outcome_description = conversation.outcome.description if conversation.outcome else ''
    
    with st.expander("📝 Conversation Details", expanded=False):
        st.write(f"**Outcome:** {outcome_name}")
        if outcome_description:
            st.write(f"**Description:** {outcome_description}")
        st.write(f"**Total Messages:** {len(conversation.messages)}")
    
    # Display messages
    for i, message in enumerate(conversation.messages):
        sender = message.sender.value
        content = message.content
        timestamp = message.timestamp.isoformat() if message.timestamp else ''
        
        # Create columns for chat-like display
        if sender == 'customer':
            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                st.markdown(
                    f"""
                    <div style="background-color: #e1f5fe; padding: 10px; border-radius: 10px; margin: 5px 0;">
                        <strong>🙋‍♀️ Customer:</strong><br>
                        {content}
                        <br><small style="color: #666;">📅 {timestamp}</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:  # agent
            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                st.markdown(
                    f"""
                    <div style="background-color: #f3e5f5; padding: 10px; border-radius: 10px; margin: 5px 0;">
                        <strong>👨‍💼 Agent:</strong><br>
                        {content}
                        <br><small style="color: #666;">📅 {timestamp}</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


def create_outcome_pie_chart(outcome_distribution, outcome_colors, title):
    """Create a pie chart for outcome distribution."""
    if outcome_distribution.outcome_counts or outcome_distribution.conversations_without_outcome > 0:
        outcome_data = [
            {'outcome': outcome, 'count': count}
            for outcome, count in outcome_distribution.outcome_counts.items()
        ]
        outcome_data.append({
            'outcome': 'unknown',
            'count': outcome_distribution.conversations_without_outcome
        })

        outcomes_df = pd.DataFrame(outcome_data)
        fig_orig = px.pie(outcomes_df, values='count', names='outcome',
                          color_discrete_map=outcome_colors,
                          color='outcome', title=title)
        st.plotly_chart(fig_orig, use_container_width=True)


def get_openai_models() -> dict[str, list[str]]:
    """Get available OpenAI models organized by category."""
    return {
        "GPT Models": [
            "gpt-4.1-2025-04-14",
            "gpt-4.1-mini-2025-04-14",
            "gpt-4.1-nano-2025-04-14",
            "gpt-4.5-preview-2025-02-27",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-2024-11-20",
        ],
        "Reasoning Models": [
            "o1-2024-12-17",
            "o1-pro-2025-03-19",
            "o1-mini-2024-09-12",
            "o3-pro-2025-06-10",
            "o3-2025-04-16",
            "o3-mini-2025-01-31",
            "o4-mini-2025-04-16"
        ],
        "Embedding Models": [
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002"
        ]
    }


def format_results_for_download(result: SimulationSessionResult, filename_prefix: str = "simulation_results") -> tuple[str, str]:
    """Format simulation results for download."""
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    
    # Convert result to JSON string
    result_dict = converter.unstructure(result)

    json_str = json.dumps(result_dict, indent=2, ensure_ascii=False, default=str)

    return json_str, filename


def validate_api_key() -> bool:
    """Validate that OpenAI API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("🔑 OpenAI API Key Required")
        st.markdown("""
        Please set your OpenAI API key as an environment variable:
        ```bash
        export OPENAI_API_KEY='your-api-key-here'
        ```
        Or add it to a `.env` file in your project root.
        """)
        return False
    return True


def show_simulation_progress(current: int, total: int, description: str = "Running simulation"):
    """Show simulation progress with a progress bar."""
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"{description}... ({current}/{total})")


def extract_unique_outcomes(conversations: list[Conversation]) -> list[dict]:
    """Extract unique outcomes from conversations."""
    unique_outcomes = {}
    
    for conversation in conversations:
        outcome = conversation.outcome
        if outcome:
            outcome_name = outcome.name
            if outcome_name and outcome_name not in unique_outcomes:
                unique_outcomes[outcome_name] = {
                    'name': outcome.name,
                    'description': outcome.description
                }
    
    return list(unique_outcomes.values())
