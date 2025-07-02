# Conversation Simulator Results Analyzer

A Streamlit web application for analyzing and visualizing RAG simulation results.

## Features

### 📊 Statistics Dashboard
- Session overview with duration, conversation counts, and metadata
- Outcome distribution comparison with pie charts
- Message length distribution analysis with histograms
- Summary statistics for original vs simulated conversations
- Adversarial Evaluation: Quality assessment using AI-powered distinction testing

### 🔍 Browse Conversations
- Interactive filtering by outcome and message count
- Chat-like conversation viewer with metadata
- Select and view individual conversations

### 🆚 Compare Conversations
- Side-by-side comparison of original vs simulated conversations
- Metrics comparison: message counts, outcomes, and message lengths

## Installation & Setup

Required packages: `streamlit`, `pandas`, `plotly`

```bash
# Using Poetry (recommended)
poetry install --with streamlit

# Or using pip
pip install streamlit pandas plotly
```

## Running the App

```bash
# Using Poetry
poetry run streamlit run streamlit/Conversation Simulator Results Analyzer.py

# Or directly
streamlit run streamlit/Conversation Simulator Results Analyzer.py
```

## Usage

1. Upload your simulation results JSON file using the sidebar uploader
2. Navigate between Statistics, Browse, and Compare tabs
3. **Statistics Dashboard**: View comprehensive analysis including:
   - Session overview and metadata
   - Outcome distribution comparisons with pie charts
   - Message length distribution histograms
   - Summary statistics table
   - Adversarial evaluation quality assessment
4. **Browse Conversations**: Use filters to find specific conversations by outcome and message count
5. **Compare Tool**: Select conversations for side-by-side analysis with detailed metrics

## Data Format

Expected JSON structure (created by the simulator):
- `session_name`, `session_description`
- `started_at`, `completed_at` (ISO timestamps)
- `original_conversations`, `simulated_conversations` arrays
- `analysis_result`: Contains comprehensive analysis data including:
  - `outcome_comparison`: Original vs simulated outcome distributions
  - `message_distribution_comparison`: Message length statistics and distributions
  - `adversarial_evaluation`: AI-powered quality assessment results

Each conversation includes:
- `id`: Unique identifier
- `conversation`: Contains `messages` array and `outcome`
- Simulated conversations: `scenario_id` and `original_conversation_id`

## Troubleshooting

**Common Issues:**
- **Import Errors**: Install required packages (`poetry install`)
- **JSON Format Errors**: Check your file follows the expected data format
- **Empty Data**: Ensure both original and simulated conversations are present
- **Performance**: Large datasets may be slow; consider filtering data
