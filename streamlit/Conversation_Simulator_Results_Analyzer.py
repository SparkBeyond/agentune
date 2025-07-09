"""
💬 Conversation Simulator Results Analyzer

A Streamlit app for analyzing and visualizing RAG simulation results with file upload functionality.
This is the main page of the Conversation Simulator Streamlit application.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Import simulation result model
from conversation_simulator.models.results import SimulationSessionResult

# Import helper functions
from helper import (
    load_simulation_results, extract_conversation_data, select_from_dataframe,
    display_conversation, create_outcome_pie_chart
)

# Set page config
st.set_page_config(
    page_title="Conversation Simulator",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def create_statistics_dashboard(original_df: pd.DataFrame, simulated_df: pd.DataFrame, results: SimulationSessionResult):
    """Create comprehensive statistics dashboard."""
    st.header("📊 Simulation Statistics")
    
    # Extract analysis results
    analysis_result = results.analysis_result
    
    # Session overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Session Name",
            results.session_name,
        )
    
    with col2:
        duration = "Unknown"
        if results.started_at and results.completed_at:
            try:
                start = results.started_at
                end = results.completed_at
                duration = str(end - start).split('.')[0]  # Remove microseconds
            except (ValueError, TypeError):
                duration = "Unknown"
        st.metric("Duration", duration)
    
    with col3:
        st.metric("Original Conversations", len(original_df))
    
    with col4:
        st.metric("Simulated Conversations", len(simulated_df))

    # Description
    st.subheader("📝 Session Description")
    st.write(results.session_description)

    # Outcome distribution comparison
    st.subheader("🎯 Outcome Distribution Comparison")
    
    # Use pre-calculated outcome distributions from analysis_result
    outcome_comparison = analysis_result.outcome_comparison
    
    if outcome_comparison:
        # Create consistent color mapping for outcomes across all distributions
        all_outcomes = set(['unknown'])  # Ensure 'unknown' is always included
        
        # Collect outcomes from all distributions
        distributions = [
            ('Original Distribution', outcome_comparison.original_distribution),
            ('Simulated Distribution', outcome_comparison.simulated_distribution),
            ('Original With Predicted Outcomes', outcome_comparison.original_with_predicted_outcomes)
        ]
        
        for dist_name, dist_data in distributions:
            if dist_data.outcome_counts:
                all_outcomes.update(dist_data.outcome_counts.keys())
        
        colors = px.colors.qualitative.Light24
        outcome_colors = {outcome: colors[i % len(colors)] for i, outcome in enumerate(sorted(all_outcomes))}
        
        # Create columns for the distributions
        num_distributions = len(distributions)
        cols = st.columns(num_distributions)
        
        for i, (dist_name, dist_data) in enumerate(distributions):
            with cols[i]:
                create_outcome_pie_chart(dist_data, outcome_colors, dist_name)
    else:
        st.info("No outcome comparison data available.")
    
    # Message length distribution
    st.subheader("📏 Message Length Distribution")
    
    # Use pre-calculated message distribution statistics from analysis_result
    message_comparison = analysis_result.message_distribution_comparison
    
    # Create data for histogram using pre-calculated distributions
    histogram_data = []
    
    # Process original and simulated stats
    stats_mapping = [
        ('Original', message_comparison.original_stats),
        ('Simulated', message_comparison.simulated_stats)
    ]
    
    for type_name, msg_stats in stats_mapping:
        if msg_stats.message_count_distribution:
            for msg_count, frequency in msg_stats.message_count_distribution.items():
                for _ in range(frequency):
                    histogram_data.append({'num_messages': int(msg_count), 'type': type_name})
    
    if histogram_data:
        combined_df = pd.DataFrame(histogram_data)
        
        # Create dynamic color mapping for all types
        unique_types = combined_df['type'].unique()
        colors = px.colors.qualitative.Set1
        dynamic_color_map = {type_name: colors[i % len(colors)] for i, type_name in enumerate(unique_types)}
        
        fig_hist = px.histogram(
            combined_df,
            x='num_messages',
            color='type',
            nbins=20,
            title="Distribution of Conversation Lengths",
            labels={'num_messages': 'Number of Messages', 'count': 'Frequency'},
            barmode='overlay',
            color_discrete_map=dynamic_color_map
        )
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No message length distribution data available.")
    
    # Summary statistics table
    st.subheader("📈 Summary Statistics")
    
    # Use pre-calculated statistics
    stats_data = []
    
    if outcome_comparison and message_comparison:
        # Create statistics for original and simulated conversations
        summary_stats_mapping = [
            ('Original', outcome_comparison.original_distribution, message_comparison.original_stats),
            ('Simulated', outcome_comparison.simulated_distribution, message_comparison.simulated_stats),
            ('Original With Predicted Outcomes', outcome_comparison.original_with_predicted_outcomes, message_comparison.original_stats)
        ]
        
        for type_name, dist_data, msg_stats in summary_stats_mapping:
            if dist_data and msg_stats:
                # Find most common outcome
                most_common_outcome = 'N/A'
                if dist_data.outcome_counts:
                    most_common_outcome = max(dist_data.outcome_counts.items(), key=lambda x: x[1])[0]
                
                stats_data.append({
                    'Type': type_name,
                    'Count': dist_data.total_conversations,
                    'Avg Messages': f"{msg_stats.mean_messages:.1f}",
                    'Min Messages': msg_stats.min_messages,
                    'Max Messages': msg_stats.max_messages,
                    'Std Dev Messages': f"{msg_stats.std_dev_messages:.1f}",
                    'Most Common Outcome': most_common_outcome
                })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("No statistics data available.")
    
    # Adversarial Evaluation
    st.subheader("🎭 Adversarial Evaluation")
    
    # Use pre-calculated adversarial evaluation from analysis_result
    adversarial_eval = analysis_result.adversarial_evaluation
    
    if adversarial_eval and adversarial_eval.total_pairs_evaluated > 0:
        col1, col2, col3 = st.columns(3)
        
        total_pairs = adversarial_eval.total_pairs_evaluated
        correct_identifications = adversarial_eval.correct_identifications
        accuracy = (correct_identifications / total_pairs * 100) if total_pairs > 0 else 0
        
        with col1:
            # Determine quality assessment (ideal is ~50% - indistinguishable like a coin toss)
            if accuracy >= 75:
                quality = "🔴 Simulated looks fake"
                quality_desc = "High accuracy means simulated conversations are easily distinguishable - they don't look realistic enough"
            elif accuracy >= 60:
                quality = "🟡 Moderately distinguishable"
                quality_desc = "Above-ideal accuracy suggests simulated conversations have noticeable differences from originals"
            elif accuracy >= 40:
                quality = "🟢 Indistinguishable (ideal)"
                quality_desc = "Accuracy near 50% is perfect - simulated conversations are indistinguishable from originals (coin toss)"
            elif accuracy >= 25:
                quality = "🟡 Simulated looks too good"
                quality_desc = "Below-ideal accuracy suggests simulated conversations may look more realistic than originals - investigate"
            else:
                quality = "🔴 Simulated preferred over original"
                quality_desc = "Very low accuracy suggests systematic bias - adversarial consistently prefers simulated over original conversations"
            
            st.metric(
                "Quality Assessment",
                quality
            )
            st.info(quality_desc)

        with col3:
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
    else:
        st.info("No adversarial evaluation data available or evaluation was not performed.")


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
                        delta=f"{orig_outcome} vs {sim_outcome}"
                    )
                
                with comparison_col3:
                    # Calculate average message length for both conversations
                    orig_msgs = selected_orig['conversation_data'].messages
                    sim_msgs = selected_sim['conversation_data'].messages

                    orig_avg_len = np.mean([len(msg.content) for msg in orig_msgs]) if orig_msgs else 0
                    sim_avg_len = np.mean([len(msg.content) for msg in sim_msgs]) if sim_msgs else 0
                    
                    st.metric(
                        "Avg Message Length",
                        f"{sim_avg_len:.0f} chars",
                        delta=f"{sim_avg_len - orig_avg_len:.0f}"
                    )
        else:
            st.info("Need both original and simulated conversations for comparison.")


if __name__ == "__main__":
    main()
