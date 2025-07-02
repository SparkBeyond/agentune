"""
Helper functions for the Conversation Simulator Streamlit apps.

Common utilities shared across different pages of the application.
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
from typing import Dict, Optional, Tuple, Any


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


def select_from_dataframe(df: pd.DataFrame, table_name: str, multi_rows: bool = False) -> Tuple[Any, Any]:
    """Select conversations from dataframe with filtering."""
    
    if df.empty:
        st.warning(f"No {table_name} available.")
        return ([] if multi_rows else None), ([] if multi_rows else None)
    
    # Add filters
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
    
    # Apply filters
    filtered_df = df.copy()
    if selected_outcome != 'All':
        filtered_df = filtered_df[filtered_df['outcome'] == selected_outcome]
    
    filtered_df = filtered_df[
        (filtered_df['num_messages'] >= message_range[0]) & (filtered_df['num_messages'] <= message_range[1])
    ]
    
    if filtered_df.empty:
        st.warning("No conversations match the selected filters.")
        return ([] if multi_rows else None), ([] if multi_rows else None)
    
    # Display selection table
    st.subheader(f"📋 Select {table_name}")
    
    # Show summary columns for selection
    display_df = filtered_df[['id', 'outcome', 'num_messages', 'first_message']].copy()
    
    selection_result = st.dataframe(
        data=display_df,
        use_container_width=True,
        key=f"select_{table_name}",
        on_select="rerun",
        selection_mode="single-row" if not multi_rows else "multi-row"
    )
    
    selected_rows = selection_result.selection['rows']  # type: ignore
    
    if selected_rows:
        if not multi_rows:
            idx = selected_rows[0]
            row = filtered_df.iloc[idx]
            return row.to_dict(), int(row.name)
        else:
            # Multi selection
            out_list, id_list = [], []
            for idx in selected_rows:
                row = filtered_df.iloc[idx]
                out_list.append(row.to_dict())
                id_list.append(int(row.name))
            return out_list, id_list
    
    # Nothing selected
    st.info("Select a row to view conversation details.")
    return ([] if multi_rows else None), ([] if multi_rows else None)


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
