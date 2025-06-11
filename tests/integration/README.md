# Integration Tests

This directory contains integration tests that make real API calls to OpenAI

## Overview

Integration tests are **excluded from default pytest runs** to avoid:
- Requiring API keys for basic development
- Incurring API costs during regular testing

## Running Integration Tests

### Prerequisites

Set up API credentials using one of these methods:

**Method 1: .env file (recommended for development)**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key
OPENAI_API_KEY=your-openai-api-key-here
```

**Method 2: Environment variable**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Running the Tests

**Run integration tests:**
```bash
poetry run pytest -m integration
```

**Run with verbose output:**
```bash
poetry run pytest -m integration -v
```

**Run specific integration test:**
```bash
poetry run pytest tests/integration/test_zero_shot_agent.py::TestZeroShotAgentIntegration::test_agent_intent_customer_initiates -m integration
```
