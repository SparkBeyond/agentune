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
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List


def load_simulation_results(uploaded_file) -> Optional[Dict]:
    """Load simulation results from uploaded JSON file."""
    try:
        content = uploaded_file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        return dict(json.loads(content))
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON file: {e}")
        return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def load_conversation_data(uploaded_file) -> Optional[List[Dict[str, Any]]]:
    """Load conversation data from uploaded JSON file."""
    try:
        content = uploaded_file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        data = json.loads(content)
        
        # Handle different formats
        if isinstance(data, dict) and 'conversations' in data:
            conversations = data['conversations']
            if isinstance(conversations, list):
                return conversations
        elif isinstance(data, list):
            return data
        
        st.error("Invalid conversation data format. Expected list of conversations or dict with 'conversations' key.")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON file: {e}")
        return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def extract_conversation_data(results: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract conversation data into DataFrames for analysis."""
    
    # Extract original conversations
    original_data = []
    for conv in results.get('original_conversations', []):
        conv_data = conv['conversation']
        # Handle outcome safely
        outcome = conv_data.get('outcome')
        outcome_name = outcome.get('name', 'unknown') if outcome else 'unknown'
        
        original_data.append({
            'id': conv['id'],
            'type': 'Original',
            'num_messages': len(conv_data['messages']),
            'outcome': outcome_name,
            'first_message': conv_data['messages'][0]['content'][:100] + "..." if conv_data['messages'] else "",
            'conversation_data': conv_data
        })
    
    # Extract simulated conversations
    simulated_data = []
    for conv in results.get('simulated_conversations', []):
        conv_data = conv['conversation']
        # Handle outcome safely
        outcome = conv_data.get('outcome')
        outcome_name = outcome.get('name', 'unknown') if outcome else 'unknown'
        
        simulated_data.append({
            'id': conv['id'],
            'type': 'Simulated',
            'scenario_id': conv.get('scenario_id', 'unknown'),
            'original_id': conv.get('original_conversation_id', 'unknown'),
            'num_messages': len(conv_data['messages']),
            'outcome': outcome_name,
            'first_message': conv_data['messages'][0]['content'][:100] + "..." if conv_data['messages'] else "",
            'conversation_data': conv_data
        })
    
    original_df = pd.DataFrame(original_data)
    simulated_df = pd.DataFrame(simulated_data)
    
    return original_df, simulated_df


def conversations_to_dataframe(conversations: List[Dict]) -> pd.DataFrame:
    """Convert conversation data to DataFrame for display and selection."""
    data = []
    for i, conv in enumerate(conversations):
        # Handle outcome safely
        outcome = conv.get('outcome')
        outcome_name = outcome.get('name', 'unknown') if outcome else 'unknown'
        
        data.append({
            'index': i,
            'id': conv.get('id', f'conversation_{i}'),
            'num_messages': len(conv.get('messages', [])),
            'outcome': outcome_name,
            'first_message': conv.get('messages', [{}])[0].get('content', '')[:100] + "..." if conv.get('messages') else "",
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
        message_range = st.slider(
            "Filter by message count",
            min_messages,
            max_messages,
            (min_messages, max_messages),
            key=f"messages_{table_name}"
        )
    
    # Check if filters changed and clear random selection if so
    filter_state_key = f"filter_state_{table_name}"
    current_filter_state = (selected_outcome, message_range)
    
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
    
    # Reset index to ensure continuous 0-based indexing
    filtered_df = filtered_df.reset_index(drop=True)
    
    return filtered_df


def show_random_selection_controls(length: int, table_name: str) -> Optional[List[int]]:
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
) -> Tuple[Any, Any]:
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


def display_conversation(conversation_data: Dict, title: str = "Conversation"):
    """Display a conversation in a chat-like format."""
    
    st.subheader(f"💬 {title}")
    
    # Conversation metadata
    with st.expander("📝 Conversation Details", expanded=False):
        outcome = conversation_data.get('outcome', {})
        outcome_name = outcome.get('name', 'unknown') if outcome else 'unknown'
        st.write(f"**Outcome:** {outcome_name}")
        if outcome and outcome.get('description'):
            st.write(f"**Description:** {outcome['description']}")
        st.write(f"**Total Messages:** {len(conversation_data.get('messages', []))}")
    
    # Display messages
    messages = conversation_data.get('messages', [])
    
    for i, message in enumerate(messages):
        sender = message.get('sender', 'unknown')
        content = message.get('content', '')
        timestamp = message.get('timestamp', '')
        
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
    if outcome_distribution.get('outcome_counts') or outcome_distribution.get('conversations_without_outcome', 0) > 0:
        outcome_data = [
            {'outcome': outcome, 'count': count, 'color': outcome_colors[outcome]}
            for outcome, count in outcome_distribution.get('outcome_counts', {}).items()
        ]
        outcome_data.append({
            'outcome': 'No Outcome',
            'count': outcome_distribution['conversations_without_outcome'],
            'color': outcome_colors['No Outcome']
        })

        outcomes_df = pd.DataFrame(outcome_data)
        fig_orig = px.pie(outcomes_df, values='count', names='outcome', color='color', title=title)
        st.plotly_chart(fig_orig, use_container_width=True)


def get_openai_models() -> Dict[str, List[str]]:
    """Get available OpenAI models organized by category."""
    return {
        "GPT Models": [
            "gpt-4.1-2025-04-14",
            "gpt-4.1-mini-2025-04-14",
            "gpt-4.1-nano-2025-04-14",
            "gpt-4.5-preview-2025-02-27",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
        ],
        "Thinking Models": [
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


def format_results_for_download(result, filename_prefix: str = "simulation_results") -> Tuple[str, str]:
    """Format simulation results for download."""
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    
    # Convert result to JSON string
    if hasattr(result, '__dict__'):
        # Handle custom objects by converting to dict
        result_dict = result.__dict__
    else:
        result_dict = result
    
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


def extract_unique_outcomes(conversations: List[Dict]) -> List[Dict]:
    """Extract unique outcomes from conversations."""
    unique_outcomes = {}
    
    for conversation in conversations:
        outcome = conversation.get('outcome')
        if outcome:
            outcome_name = outcome.get('name')
            if outcome_name and outcome_name not in unique_outcomes:
                unique_outcomes[outcome_name] = outcome
    
    return list(unique_outcomes.values())
