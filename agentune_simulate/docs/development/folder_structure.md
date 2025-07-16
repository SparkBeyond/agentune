# Conversation Simulator - Folder Structure

This document outlines the high-level folder structure for the Conversation Simulator project.

## Overview

```
.
├── pyproject.toml          # Project configuration and dependencies
├── README.md               # Project overview and setup instructions
├── CODING_STANDARDS.md     # Development guidelines
├── agentune/simulate/      # Main package source code
│   ├── models/            # Core data structures
│   ├── participants/      # Customer and agent implementations
│   │   ├── customer/      # Customer participant types
│   │   └── agent/         # Agent participant types
│   ├── runners/           # Simulation execution engines
│   ├── outcome_detection/ # Conversation outcome analysis
│   └── channels/          # Communication abstractions
├── tests/                 # Test files and fixtures
│   └── fixtures/          # Sample data for testing
└── docs/                  # Documentation
    └── development/       # Developer guides and design docs
```

## Key Components

### `agentune/simulate/` - Main Package
- **`models/`** - Core data structures (Message, Conversation, Intent, Outcome, etc.)
- **`participants/`** - Customer and agent implementations with abstract base classes
- **`runners/`** - Simulation execution engines (full simulation, hybrid modes)
- **`outcome_detection/`** - Conversation outcome detection
- **`channels/`** - Communication channel abstractions (placeholder interfaces)

### `tests/` - Testing
- Unit tests for all components
- **`fixtures/`** - Sample conversation data for testing

### `docs/` - Documentation  
- **`development/`** - Technical documentation, design decisions, and guidelines
```

## Architecture Principles

- **Modular Design**: Clear separation between data models, behavior, and orchestration
- **Abstract Base Classes**: Interface definitions in `base.py` files with concrete implementations
