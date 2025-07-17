# Agentune Simulate Streamlit App

A Streamlit web application for running RAG-based conversation simulations and analyzing results.

## Features

### 🤖 Agentune Simulate Runner
- Upload conversation data and configure simulation settings
- Flexible model selection (GPT, o1, o3 models) with temperature controls
- Advanced/Basic configuration modes
- Random conversation selection with filters
- Real-time simulation progress tracking
- Download simulation results as JSON

### 📊 Results Analyzer
- Statistics dashboard with outcome distribution and message length analysis
- Browse individual conversations with interactive filtering
- Side-by-side comparison of original vs simulated conversations
- Adversarial evaluation quality assessment

## Installation & Setup

```bash
# Install dependencies
poetry install --with streamlit

# Set OpenAI API key (required for simulation)
export OPENAI_API_KEY='your-api-key-here'

# Or create a .env file in the project root
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Running the App

```bash
# Using Poetry
poetry run python -m streamlit run streamlit/Agentune_Simulate_Results_Analyzer.py

# Or directly
python -m streamlit run streamlit/Agentune_Simulate_Results_Analyzer.py
```

## Usage

### Running Simulations
1. Navigate to "Agentune Simulate Runner" page
2. Upload conversation data (JSON format)
3. Configure models and simulation parameters
4. Select conversations to simulate (manual or random)
5. Run simulation and download results

### Analyzing Results
1. Upload simulation results JSON file
2. Explore statistics, browse conversations, or compare side-by-side
3. View adversarial evaluation and quality metrics

## Data Format

**Input (for simulation):** JSON array of conversations with `messages` and `outcome`

**Output (simulation results):** Complete analysis including original conversations, simulated conversations, outcome comparisons, and adversarial evaluation
