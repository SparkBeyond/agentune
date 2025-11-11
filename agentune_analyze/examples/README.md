# Agentune Analyze Examples

This directory contains hands-on tutorials demonstrating how to use Agentune Analyze for data analysis and feature generation.

## Prerequisites

- **Python**: >=3.12, <4.0
- **Agentune Analyze**: Installed via pip or poetry
- **Jupyter Notebook**: For running the examples
- **OpenAI API Key**: Required for feature generation

## Installation

```bash
# Install agentune-analyze
pip install agentune-analyze

# Or if using poetry from the repository
poetry install --with examples

# Start Jupyter
jupyter notebook
```

This will open Jupyter in your browser where you can run the example notebooks.

**Note for Apple Silicon Mac users**: If you encounter installation errors related to architecture incompatibility, install OpenMP first: `brew install libomp`. See the [LightGBM macOS installation guide](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html) for details.

## Available Examples

### [01_getting_started.ipynb](01_getting_started.ipynb)

**Introductory walkthrough** - Start here!

Learn the fundamentals of Agentune Analyze by analyzing customer service conversations:
- Loading conversation data
- Running analysis
- Exploring discovered features and their statistics
- Generating action recommendations from conversation patterns

**Time**: ~10-15 minutes
**Dataset**: Auto insurance customer service (101 conversations)
**Prerequisites**: OpenAI API key

### [Advanced examples](examples.md)

Example code for several common usecases not covered by the walkthrough.

## Sample Data

All tutorial datasets are located in the [`data/`](data/) directory. See [data/README.md](data/README.md) for detailed schema documentation and information about using your own data.

## Utilities

The [`utils/`](utils/) directory contains helper utilities for working with tutorial results:

### Dashboard Generator

Generate interactive HTML dashboards to visualize analysis results:

```python
from utils.generate_dashboard import create_dashboard

# After running analysis
dashboard_path = create_dashboard(
    results=analyze_results,
    output_file="my_dashboard.html",
    title="My Feature Analysis"
)

print(f"Dashboard saved to: {dashboard_path}")
```

The dashboard includes:
- Target distribution visualization
- Feature performance ranking by R² (coefficient of determination)
- Sortable feature table with detailed statistics
- Interactive feature comparison tool
- Expandable detail views with lift matrices and distributions

**Note**: This is a temporary convenience utility. A full-featured dashboard solution is planned for future releases.

See the getting started notebook for a complete example.

## Learning Path

Start with [01_getting_started.ipynb](01_getting_started.ipynb) to understand the basic workflow. More advanced examples coming soon!

## Running the Examples

1. Navigate to the examples directory:
   ```bash
   cd examples
   ```

2. Start Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open `01_getting_started.ipynb` in the browser

4. Follow the instructions in the notebook

## API Key Setup

The tutorials require an OpenAI API key. You can:

1. **Set as environment variable** (recommended):
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

2. **Set in notebook**: Each notebook includes a cell for API key setup

## Documentation

- [Main README](../README.md) - Library overview
- [Architecture Guide](../docs/README.md) - Design principles and patterns
- [Data README](data/README.md) - Sample data documentation

## Questions or Issues?

Open an issue on GitHub or contact the maintainers. See the Main README for contact information.
