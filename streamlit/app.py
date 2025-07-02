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
    # Define consistent colors for Original vs Simulated across all charts
    type_color_map = {
        'Original': '#2E86C1',    # Blue
        'Simulated': '#E74C3C'    # Red
    }

    st.header("📊 Simulation Statistics")
    
    # Extract analysis results
    analysis_result = results.get('analysis_result', {})
    
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
    
    # Use pre-calculated outcome distributions from analysis_result
    outcome_comparison = analysis_result.get('outcome_comparison', {})
    original_dist = outcome_comparison.get('original_distribution', {})
    simulated_dist = outcome_comparison.get('simulated_distribution', {})
    
    # Create consistent color mapping for outcomes
    all_outcomes = set(['No Outcome'])  # Ensure 'No Outcome' is always included
    if original_dist.get('outcome_counts'):
        all_outcomes.update(original_dist['outcome_counts'].keys())
    if simulated_dist.get('outcome_counts'):
        all_outcomes.update(simulated_dist['outcome_counts'].keys())
    colors = px.colors.qualitative.Set2
    outcome_colors = {outcome: colors[i % len(colors)] for i, outcome in enumerate(sorted(all_outcomes))}
    
    col1, col2 = st.columns(2)
    
    with col1:
        _outcome_pie_chart(original_dist, outcome_colors, "Original Conversations Outcomes")

    with col2:
        _outcome_pie_chart(simulated_dist, outcome_colors, "Simulated Conversations Outcomes")
    
    # Message length distribution
    st.subheader("📏 Message Length Distribution")
    
    # Use pre-calculated message distribution statistics from analysis_result
    message_comparison = analysis_result.get('message_distribution_comparison', {})
    original_stats = message_comparison.get('original_stats', {})
    simulated_stats = message_comparison.get('simulated_stats', {})
    
    # Create data for histogram using pre-calculated distributions
    histogram_data = []
    
    if original_stats.get('message_count_distribution'):
        for msg_count, frequency in original_stats['message_count_distribution'].items():
            for _ in range(frequency):
                histogram_data.append({'num_messages': int(msg_count), 'type': 'Original'})
    
    if simulated_stats.get('message_count_distribution'):
        for msg_count, frequency in simulated_stats['message_count_distribution'].items():
            for _ in range(frequency):
                histogram_data.append({'num_messages': int(msg_count), 'type': 'Simulated'})
    
    if histogram_data:
        combined_df = pd.DataFrame(histogram_data)
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
    
    # Use pre-calculated statistics
    stats_data = []
    
    if original_stats:
        stats_data.append({
            'Type': 'Original',
            'Count': original_dist.get('total_conversations', 0),
            'Avg Messages': f"{original_stats.get('mean_messages', 0):.1f}",
            'Min Messages': original_stats.get('min_messages', 0),
            'Max Messages': original_stats.get('max_messages', 0),
            'Std Dev Messages': f"{original_stats.get('std_dev_messages', 0):.1f}",
            'Most Common Outcome': max(original_dist.get('outcome_counts', {}), key=original_dist.get('outcome_counts', {}).get, default='N/A')
        })
    
    if simulated_stats:
        stats_data.append({
            'Type': 'Simulated',
            'Count': simulated_dist.get('total_conversations', 0),
            'Avg Messages': f"{simulated_stats.get('mean_messages', 0):.1f}",
            'Min Messages': simulated_stats.get('min_messages', 0),
            'Max Messages': simulated_stats.get('max_messages', 0),
            'Std Dev Messages': f"{simulated_stats.get('std_dev_messages', 0):.1f}",
            'Most Common Outcome': max(simulated_dist.get('outcome_counts', {}), key=simulated_dist.get('outcome_counts', {}).get, default='N/A')
        })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
    
    # Adversarial Evaluation
    st.subheader("🎭 Adversarial Evaluation")
    
    # Use pre-calculated adversarial evaluation from analysis_result
    adversarial_eval = analysis_result.get('adversarial_evaluation', {})
    
    if adversarial_eval and adversarial_eval.get('total_pairs_evaluated', 0) > 0:
        col1, col2, col3 = st.columns(3)
        
        total_pairs = adversarial_eval.get('total_pairs_evaluated', 0)
        correct_identifications = adversarial_eval.get('correct_identifications', 0)
        accuracy = (correct_identifications / total_pairs * 100) if total_pairs > 0 else 0
        
        with col1:
            st.metric(
                "Total Pairs Evaluated",
                total_pairs
            )
        
        with col2:
            st.metric(
                "Correct Identifications",
                correct_identifications,
                delta=f"{accuracy:.1f}% accuracy"
            )
        
        with col3:
            # Determine quality assessment
            if accuracy >= 80:
                quality = "🔴 Easy to distinguish"
                quality_desc = "High accuracy suggests simulated conversations are easily distinguishable from originals"
            elif accuracy >= 60:
                quality = "🟡 Moderately distinguishable"
                quality_desc = "Medium accuracy suggests some differences between simulated and original conversations"
            else:
                quality = "🟢 Hard to distinguish"
                quality_desc = "Low accuracy suggests simulated conversations are very similar to originals"
            
            st.metric(
                "Quality Assessment",
                quality
            )
            st.info(quality_desc)
    else:
        st.info("No adversarial evaluation data available or evaluation was not performed.")


def _outcome_pie_chart(outcome_distribution, outcome_colors, title):
    if outcome_distribution.get('outcome_counts') or outcome_distribution.get('conversations_without_outcome', 0) > 0:
        outcome_data = [{'outcome': outcome, 'count': count, 'color': outcome_colors[outcome]} for outcome, count in outcome_distribution.get('outcome_counts', {}).items()]
        outcome_data.append({'outcome': 'No Outcome', 'count': outcome_distribution['conversations_without_outcome'], 'color': outcome_colors['No Outcome']})

        outcomes_df = pd.DataFrame(outcome_data)
        fig_orig = px.pie(outcomes_df, values='count', names='outcome', color='color', title=title)
        st.plotly_chart(fig_orig, use_container_width=True)


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
