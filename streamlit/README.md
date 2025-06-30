# Conversation Simulator Results Analyzer

A Streamlit web application for analyzing and visualizing RAG simulation results.

## Features

### 📊 Statistics Dashboard
- Session overview with duration, conversation counts, and metadata
- Outcome distribution comparison with pie charts
- Message length distribution analysis
- **Consecutive Messages Analysis**: Analyze patterns where participants send multiple messages in a row
- Summary statistics for original vs simulated conversations

### 🔍 Browse Conversations
- Interactive filtering by outcome and message count
- Chat-like conversation viewer with metadata
- Select and view individual conversations

### 🆚 Compare Conversations
- Side-by-side comparison of original vs simulated conversations
- Metrics comparison: message counts, outcomes, and message lengths

## Installation & Setup

Required packages: `streamlit`, `pandas`, `plotly`, `numpy`

```bash
# Using Poetry (recommended)
poetry install --with streamlit

# Or using pip
pip install streamlit pandas plotly numpy
```

## Running the App

```bash
# Using Poetry
poetry run streamlit run streamlit/app.py

# Or directly
streamlit run streamlit/app.py
```

## Usage

1. Upload your simulation results JSON file using the sidebar uploader
2. Navigate between Statistics, Browse, and Compare tabs
3. Use filters to find specific conversations
4. Click table rows to view conversations
5. Select conversations in Compare tab for side-by-side analysis

## Data Format

Expected JSON structure:
- `session_name`, `session_description`
- `started_at`, `completed_at` (ISO timestamps)
- `original_conversations`, `simulated_conversations` arrays

Each conversation includes:
- `id`: Unique identifier
- `conversation`: Contains `messages` array and `outcome`
- Simulated conversations: `scenario_id` and `original_conversation_id`

## Consecutive Messages Analysis

Analyzes patterns where the same participant sends multiple messages consecutively. Provides:
- Overview metrics and distribution charts
- Comparison between original and simulated patterns
- Insights into conversation dynamics and communication styles

## Troubleshooting

**Common Issues:**
- **Import Errors**: Install required packages (`poetry install`)
- **JSON Format Errors**: Check your file follows the expected data format
- **Empty Data**: Ensure both original and simulated conversations are present
- **Performance**: Large datasets may be slow; consider filtering data
