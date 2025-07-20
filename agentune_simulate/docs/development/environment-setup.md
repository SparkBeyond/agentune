# Development Environment Setup

## Prerequisites

- Python 3.12+ 
- Poetry ([installation guide](https://python-poetry.org/docs/#installation))

## Quick Start

```bash
git clone https://github.com/SparkBeyond/agentune.git
cd agentune_simulate
```

Setup a python environment, and install dependencies:

```bash
poetry install
```

## Environment Variables (Optional)

For integration tests and API access:

```bash
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=your-openai-api-key-here
```

## Development Commands

### Testing

```bash
# Unit tests (default)
poetry run pytest

# Integration tests (requires OpenAI API key)
poetry run pytest -m integration

# Run specific test file
poetry run pytest tests/test_example.py

# Verbose output
poetry run pytest -v
```

### Code Quality

```bash
# Linting
poetry run ruff check .

# Auto-fix issues when possible
poetry run ruff check --fix .

# Type checking
poetry run mypy .
```

---

For detailed coding standards and architectural guidelines, see the other files in `docs/development/`.