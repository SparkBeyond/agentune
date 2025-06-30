"""
Conversation Simulator Results Analyzer

A Streamlit app for analyzing and visualizing RAG simulation results with file upload functionality.
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
from datetime import datetime
import numpy as np
from typing import Dict, Optional, Tuple, Any

# Set page config
st.set_page_config(
    page_title="Conversation Simulator Results",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)


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


def create_statistics_dashboard(original_df: pd.DataFrame, simulated_df: pd.DataFrame, results: Dict):
    """Create comprehensive statistics dashboard."""
    
    st.header("📊 Simulation Statistics")
    
    # Calculate consecutive message statistics
    original_with_stats = calculate_consecutive_stats_for_dataframe(original_df)
    simulated_with_stats = calculate_consecutive_stats_for_dataframe(simulated_df)
    
    # Session overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Session Name",
            results.get('session_name', 'Unknown'),
        )
    
    with col2:
        duration = "Unknown"
        if results.get('started_at') and results.get('completed_at'):
            try:
                start = datetime.fromisoformat(results['started_at'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(results['completed_at'].replace('Z', '+00:00'))
                duration = str(end - start).split('.')[0]  # Remove microseconds
            except (ValueError, TypeError):
                duration = "Unknown"
        st.metric("Duration", duration)
    
    with col3:
        st.metric("Original Conversations", len(original_df))
    
    with col4:
        st.metric("Simulated Conversations", len(simulated_df))
    
    # Outcome distribution comparison
    st.subheader("🎯 Outcome Distribution Comparison")
    
    # Create consistent color mapping for outcomes
    all_outcomes = set() 
    if not original_df.empty:
        all_outcomes.update(original_df['outcome'].unique())
    if not simulated_df.empty:
        all_outcomes.update(simulated_df['outcome'].unique())
    
    # Create a consistent color palette for all outcomes
    colors = px.colors.qualitative.Set2
    outcome_colors = {outcome: colors[i % len(colors)] for i, outcome in enumerate(sorted(all_outcomes))}
    
    # Define consistent colors for Original vs Simulated across all charts
    type_color_map = {
        'Original': '#2E86C1',    # Blue
        'Simulated': '#E74C3C'    # Red
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not original_df.empty:
            orig_outcomes = original_df.groupby('outcome').size().reset_index(name='count')
            orig_colors = [outcome_colors[outcome] for outcome in orig_outcomes['outcome']]
            
            fig_orig = px.pie(
                orig_outcomes,
                values='count',
                names='outcome',
                title="Original Conversations Outcomes",
                color_discrete_sequence=orig_colors
            )
            st.plotly_chart(fig_orig, use_container_width=True)
    
    with col2:
        if not simulated_df.empty:
            sim_outcomes = simulated_df.groupby('outcome').size().reset_index(name='count')
            sim_colors = [outcome_colors[outcome] for outcome in sim_outcomes['outcome']]
            
            fig_sim = px.pie(
                sim_outcomes,
                values='count',
                names='outcome',
                title="Simulated Conversations Outcomes",
                color_discrete_sequence=sim_colors
            )
            st.plotly_chart(fig_sim, use_container_width=True)
    
    # Message length distribution
    st.subheader("📏 Message Length Distribution")
    
    combined_df = pd.concat([
        original_df[['num_messages', 'type']],
        simulated_df[['num_messages', 'type']]
    ], ignore_index=True)
    
    if not combined_df.empty:
        fig_hist = px.histogram(
            combined_df,
            x='num_messages',
            color='type',
            nbins=20,
            title="Distribution of Conversation Lengths",
            labels={'num_messages': 'Number of Messages', 'count': 'Frequency'},
            barmode='overlay',
            color_discrete_map=type_color_map
        )
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Summary statistics table
    st.subheader("📈 Summary Statistics")
    
    stats_data = []
    for df, label in [(original_df, 'Original'), (simulated_df, 'Simulated')]:
        if not df.empty:
            stats_data.append({
                'Type': label,
                'Count': len(df),
                'Avg Messages': f"{df['num_messages'].mean():.1f}",
                'Min Messages': df['num_messages'].min(),
                'Max Messages': df['num_messages'].max(),
                'Most Common Outcome': df['outcome'].mode().iloc[0] if not df['outcome'].mode().empty else 'N/A'
            })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
    
    # Consecutive Messages Analysis
    st.subheader("🔄 Consecutive Messages Analysis")
    
    # Calculate consecutive message statistics
    original_with_stats = calculate_consecutive_stats_for_dataframe(original_df)
    simulated_with_stats = calculate_consecutive_stats_for_dataframe(simulated_df)
    
    if not original_with_stats.empty or not simulated_with_stats.empty:
        # Consecutive message metrics overview
        st.markdown("##### Overview of Consecutive Message Patterns")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate aggregate statistics
        orig_stats = {}
        sim_stats = {}
        
        if not original_with_stats.empty:
            orig_stats = {
                'total_sequences': original_with_stats['total_consecutive_sequences'].sum(),
                'avg_customer_consecutive': original_with_stats['avg_consecutive_customer'].mean(),
                'avg_agent_consecutive': original_with_stats['avg_consecutive_agent'].mean(),
                'max_customer_consecutive': original_with_stats['max_consecutive_customer'].max(),
                'max_agent_consecutive': original_with_stats['max_consecutive_agent'].max()
            }
        
        if not simulated_with_stats.empty:
            sim_stats = {
                'total_sequences': simulated_with_stats['total_consecutive_sequences'].sum(),
                'avg_customer_consecutive': simulated_with_stats['avg_consecutive_customer'].mean(),
                'avg_agent_consecutive': simulated_with_stats['avg_consecutive_agent'].mean(),
                'max_customer_consecutive': simulated_with_stats['max_consecutive_customer'].max(),
                'max_agent_consecutive': simulated_with_stats['max_consecutive_agent'].max()
            }
        
        with col1:
            st.metric(
                "Total Consecutive Sequences",
                f"Sim: {sim_stats.get('total_sequences', 0)}" if sim_stats else "N/A",
                delta=f"Orig: {orig_stats.get('total_sequences', 0)}" if orig_stats else None
            )
        
        with col2:
            st.metric(
                "Avg Customer Consecutive",
                f"{sim_stats.get('avg_customer_consecutive', 0):.1f}" if sim_stats else "N/A",
                delta=f"{sim_stats.get('avg_customer_consecutive', 0) - orig_stats.get('avg_customer_consecutive', 0):.1f}" if (sim_stats and orig_stats) else None
            )
        
        with col3:
            st.metric(
                "Avg Agent Consecutive",
                f"{sim_stats.get('avg_agent_consecutive', 0):.1f}" if sim_stats else "N/A",
                delta=f"{sim_stats.get('avg_agent_consecutive', 0) - orig_stats.get('avg_agent_consecutive', 0):.1f}" if (sim_stats and orig_stats) else None
            )
        
        with col4:
            st.metric(
                "Max Consecutive (Any)",
                f"Sim: {max(sim_stats.get('max_customer_consecutive', 0), sim_stats.get('max_agent_consecutive', 0))}" if sim_stats else "N/A",
                delta=f"Orig: {max(orig_stats.get('max_customer_consecutive', 0), orig_stats.get('max_agent_consecutive', 0))}" if orig_stats else None
            )
        
        # Consecutive message distribution charts
        st.markdown("##### Distribution of Consecutive Message Lengths")
        
        # Prepare data for plotting
        consecutive_data = []
        
        for df, conv_type in [(original_with_stats, 'Original'), (simulated_with_stats, 'Simulated')]:
            if not df.empty:
                for _, row in df.iterrows():
                    conversation_data = row['conversation_data']
                    stats = analyze_consecutive_messages(conversation_data)
                    
                    for seq in stats['consecutive_sequences_details']:
                        consecutive_data.append({
                            'Type': conv_type,
                            'Sender': seq['sender'].title(),
                            'Consecutive_Count': seq['count'],
                            'Conversation_ID': row['id']
                        })
        
        if consecutive_data:
            consecutive_df = pd.DataFrame(consecutive_data)
            
            # Create histogram of consecutive message lengths
            fig_consecutive = px.histogram(
                consecutive_df,
                x='Consecutive_Count',
                color='Type',
                facet_col='Sender',
                title="Distribution of Consecutive Message Lengths by Sender and Type",
                labels={'Consecutive_Count': 'Consecutive Messages', 'count': 'Frequency'},
                nbins=int(min(20, consecutive_df['Consecutive_Count'].max())),
                color_discrete_map=type_color_map
            )
            fig_consecutive.update_layout(bargap=0.1)
            st.plotly_chart(fig_consecutive, use_container_width=True)
            
            # Summary table for consecutive messages
            st.markdown("##### Consecutive Messages Summary Table")
            
            summary_data = []
            for conv_type in ['Original', 'Simulated']:
                type_data = consecutive_df[consecutive_df['Type'] == conv_type]
                if not type_data.empty:
                    for sender in ['Customer', 'Agent']:
                        sender_data = type_data[type_data['Sender'] == sender]
                        if not sender_data.empty:
                            summary_data.append({
                                'Type': conv_type,
                                'Sender': sender,
                                'Total Sequences': len(sender_data),
                                'Avg Length': f"{sender_data['Consecutive_Count'].mean():.1f}",
                                'Max Length': sender_data['Consecutive_Count'].max(),
                                'Min Length': sender_data['Consecutive_Count'].min()
                            })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("No consecutive message sequences found in the conversations.")
    else:
        st.info("No conversation data available for consecutive message analysis.")


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
    mode = "single-row" if not multi_rows else "multi-row"
    st.subheader(f"📋 Select {table_name}")
    
    # Show summary columns for selection
    display_df = filtered_df[['id', 'outcome', 'num_messages', 'first_message']].copy()
    
    selection_result = st.dataframe(
        data=display_df,
        use_container_width=True,
        key=f"select_{table_name}",
        on_select="rerun",
        selection_mode=mode
    )  # type: ignore[call-overload]
    
    selected_rows = selection_result.selection['rows']
    
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


def analyze_consecutive_messages(conversation_data: Dict) -> Dict[str, Any]:
    """Analyze consecutive messages from the same sender in a conversation.
    
    Args:
        conversation_data: Dictionary containing conversation messages
        
    Returns:
        Dictionary with consecutive message statistics
    """
    messages = conversation_data.get('messages', [])
    if len(messages) < 2:
        return {
            'total_consecutive_sequences': 0,
            'customer_consecutive_sequences': 0,
            'agent_consecutive_sequences': 0,
            'max_consecutive_customer': 0,
            'max_consecutive_agent': 0,
            'avg_consecutive_customer': 0,
            'avg_consecutive_agent': 0,
            'consecutive_sequences_details': []
        }
    
    consecutive_sequences = []
    current_sender = messages[0].get('sender', 'unknown')
    current_count = 1
    
    # Analyze consecutive messages
    for i in range(1, len(messages)):
        sender = messages[i].get('sender', 'unknown')
        
        if sender == current_sender:
            current_count += 1
        else:
            # End of consecutive sequence (only count if > 1 message)
            if current_count > 1:
                consecutive_sequences.append({
                    'sender': current_sender,
                    'count': current_count,
                    'start_index': i - current_count,
                    'end_index': i - 1
                })
            
            current_sender = sender
            current_count = 1
    
    # Don't forget the last sequence
    if current_count > 1:
        consecutive_sequences.append({
            'sender': current_sender,
            'count': current_count,
            'start_index': len(messages) - current_count,
            'end_index': len(messages) - 1
        })
    
    # Calculate statistics
    customer_sequences = [seq for seq in consecutive_sequences if seq['sender'] == 'customer']
    agent_sequences = [seq for seq in consecutive_sequences if seq['sender'] == 'agent']
    
    return {
        'total_consecutive_sequences': len(consecutive_sequences),
        'customer_consecutive_sequences': len(customer_sequences),
        'agent_consecutive_sequences': len(agent_sequences),
        'max_consecutive_customer': max([seq['count'] for seq in customer_sequences], default=0),
        'max_consecutive_agent': max([seq['count'] for seq in agent_sequences], default=0),
        'avg_consecutive_customer': np.mean([seq['count'] for seq in customer_sequences]) if customer_sequences else 0,
        'avg_consecutive_agent': np.mean([seq['count'] for seq in agent_sequences]) if agent_sequences else 0,
        'consecutive_sequences_details': consecutive_sequences
    }


def calculate_consecutive_stats_for_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate consecutive message statistics for all conversations in a dataframe.
    
    Args:
        df: DataFrame containing conversation data
        
    Returns:
        DataFrame with consecutive message statistics added
    """
    if df.empty:
        return df
    
    consecutive_stats = []
    for _, row in df.iterrows():
        stats = analyze_consecutive_messages(row['conversation_data'])
        consecutive_stats.append(stats)
    
    # Add the statistics as new columns
    stats_df = pd.DataFrame(consecutive_stats)
    result_df = df.copy()
    
    for col in stats_df.columns:
        if col != 'consecutive_sequences_details':  # Skip the detailed list
            result_df[col] = stats_df[col]
    
    return result_df


def main():
    """Main Streamlit app."""
    
    st.title("💬 Conversation Simulator Results Analyzer")
    st.markdown("Analyze and explore RAG simulation results interactively")
    
    # File uploader
    st.sidebar.header("📁 Upload Results File")
    uploaded_file = st.sidebar.file_uploader(
        "Upload simulation results JSON file",
        type=['json'],
        help="Upload the JSON file generated by the RAG simulation"
    )
    
    if uploaded_file is None:
        st.markdown("""
        ## Welcome to the Conversation Simulator Results Analyzer! 🎉
        
        To get started:
        1. **Upload your simulation results** using the file uploader in the sidebar
        2. **Explore statistics** to understand your simulation performance
        3. **Browse conversations** to see individual examples
        4. **Compare conversations** to analyze differences between original and simulated
        
        ### What you can analyze:
        - 📊 **Statistics Dashboard**: Overview of outcomes, message lengths, and performance metrics
        - 🔍 **Conversation Browser**: Filter and view individual conversations
        - 🆚 **Comparison Tool**: Side-by-side comparison of original vs simulated conversations
        
        ### Supported file format:
        - JSON files generated by the conversation simulator (e.g., `simple_simulation_results.json`)
        """)
        st.stop()
    
    # Load results
    with st.spinner("Loading simulation results..."):
        results = load_simulation_results(uploaded_file)
        if results is None:
            st.stop()
    
    st.success("✅ Successfully loaded simulation results!")
    
    # Extract data
    with st.spinner("Processing conversation data..."):
        original_df, simulated_df = extract_conversation_data(results)
    
    # Main navigation
    tab1, tab2, tab3 = st.tabs(["📊 Statistics", "🔍 Browse Conversations", "🆚 Compare Conversations"])
    
    with tab1:
        create_statistics_dashboard(original_df, simulated_df, results)
    
    with tab2:
        st.header("🔍 Browse Conversations")
        
        # Choose conversation type
        conv_type = st.radio(
            "Select conversation type to explore:",
            ["Original Conversations", "Simulated Conversations"],
            horizontal=True
        )
        
        if conv_type == "Original Conversations":
            if not original_df.empty:
                selected_conv, _ = select_from_dataframe(original_df, "Original Conversations")
                if selected_conv:
                    display_conversation(
                        selected_conv['conversation_data'],
                        f"Original Conversation: {selected_conv['id']}"
                    )
            else:
                st.info("No original conversations available.")
        
        else:  # Simulated Conversations
            if not simulated_df.empty:
                selected_conv, _ = select_from_dataframe(simulated_df, "Simulated Conversations")
                if selected_conv:
                    st.markdown("---")
                    
                    # Show simulation metadata
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Scenario ID:** {selected_conv.get('scenario_id', 'unknown')}")
                    with col2:
                        st.info(f"**Based on Original:** {selected_conv.get('original_id', 'unknown')}")
                    
                    display_conversation(
                        selected_conv['conversation_data'],
                        f"Simulated Conversation: {selected_conv['id']}"
                    )
            else:
                st.info("No simulated conversations available.")
    
    with tab3:
        st.header("🆚 Compare Original vs Simulated")
        
        if not original_df.empty and not simulated_df.empty:
            st.subheader("Select conversations to compare:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Conversation:**")
                selected_orig, _ = select_from_dataframe(original_df, "Original_Compare", multi_rows=False)
            
            with col2:
                st.markdown("**Simulated Conversation:**")
                selected_sim, _ = select_from_dataframe(simulated_df, "Simulated_Compare", multi_rows=False)
            
            if selected_orig and selected_sim:
                st.markdown("---")
                
                # Side-by-side comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    display_conversation(
                        selected_orig['conversation_data'],
                        f"Original: {selected_orig['id']}"
                    )
                
                with col2:
                    display_conversation(
                        selected_sim['conversation_data'],
                        f"Simulated: {selected_sim['id']}"
                    )
                
                # Comparison metrics
                st.subheader("📊 Comparison Metrics")
                comparison_col1, comparison_col2, comparison_col3 = st.columns(3)
                
                with comparison_col1:
                    st.metric(
                        "Message Count Difference",
                        selected_sim['num_messages'] - selected_orig['num_messages'],
                        delta=f"{selected_sim['num_messages']} vs {selected_orig['num_messages']}"
                    )
                
                with comparison_col2:
                    orig_outcome = selected_orig['outcome']
                    sim_outcome = selected_sim['outcome']
                    outcome_match = "✅ Match" if orig_outcome == sim_outcome else "❌ Different"
                    st.metric(
                        "Outcome Comparison",
                        outcome_match,
                        delta=f"{sim_outcome} vs {orig_outcome}"
                    )
                
                with comparison_col3:
                    # Calculate average message length for both conversations
                    orig_msgs = selected_orig['conversation_data']['messages']
                    sim_msgs = selected_sim['conversation_data']['messages']
                    
                    orig_avg_len = np.mean([len(msg['content']) for msg in orig_msgs]) if orig_msgs else 0
                    sim_avg_len = np.mean([len(msg['content']) for msg in sim_msgs]) if sim_msgs else 0
                    
                    st.metric(
                        "Avg Message Length",
                        f"{sim_avg_len:.0f} chars",
                        delta=f"{sim_avg_len - orig_avg_len:.0f}"
                    )
        else:
            st.info("Need both original and simulated conversations for comparison.")


if __name__ == "__main__":
    main()
