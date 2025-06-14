# Zeroshot Outcome Detection

## Overview

The `ZeroshotOutcomeDetector` analyzes conversations to determine if they've reached predefined outcomes using a language model with zero-shot capabilities.

```python
from conversation_simulator.outcome_detection import ZeroshotOutcomeDetector

# Create detector with any LangChain BaseChatModel
detector = ZeroshotOutcomeDetector(model=chat_model)

# Detect outcome
outcome = await detector.detect_outcome(conversation, intent, possible_outcomes)
```

## Key Features

- Accepts LangChain's BaseChatModel for flexible model selection
- Stateless design with no internal state between calls
- JSON-structured output with explicit reasoning and transparency
- Raises exceptions when detecting unrecognized outcomes
- Simple, focused API with minimal configuration

## Implementation Notes

- Uses the LangChain chain pattern (`PromptTemplate | Model | OutputParser`)
- Leverages Pydantic-based structured output for reliable parsing
- Separates outcome detection (chain) from outcome matching (post-processing)
- Integration tests require OpenAI API key
- Tests skip automatically when API key is missing
