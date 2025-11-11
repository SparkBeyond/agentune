# Sample Data for Agentune Analyze Tutorials

This directory contains sample datasets used in the Agentune Analyze tutorials, demonstrating conversation analysis and feature generation capabilities.

## Dataset Overview

**Domain**: Auto insurance customer service conversations
**Size**: 101 conversations with 16,823 message turns
**Purpose**: Demonstrate LLM-powered feature generation from multi-turn dialogues

## Files

### conversations.csv

Main table with conversation-level metadata and outcomes (one row per conversation).

**Size**: ~15 KB, 101 rows

**Schema**:

| Column | Type | Description |
|--------|------|-------------|
| `conversation_id` | string | Unique identifier for each conversation |
| `outcome` | string | Conversation outcome (target variable for prediction) |
| `duration_seconds` | float | Total conversation duration in seconds |

**Outcome Categories**:
- `customer not interested` (32 conversations, 31.7%)
- `process paused - customer needs to consider the offer` (28 conversations, 27.7%)
- `process paused - customer needs to collect more information` (17 conversations, 16.8%)
- `customer objections not handled` (12 conversations, 11.9%)
- `no quote - ineligible customer` (11 conversations, 10.9%)
- `buy` (1 conversation, 1.0%)

**Note**: Technical/connectivity issue outcomes have been excluded from this sample to focus on agent-controllable outcomes.

### messages.csv

Individual message turns within conversations (multiple rows per conversation).

**Size**: ~2.9 MB, 16,823 rows

**Schema**:

| Column | Type | Description |
|--------|------|-------------|
| `conversation_id` | string | Links to metadata table |
| `timestamp` | timestamp | When the message was sent (format: YYYY-MM-DD HH:MM:SS) |
| `message` | string | The message text content |
| `author` | string | Speaker role: "Agent" or "Customer" |

**Statistics**:
- Average: ~167 messages per conversation
- Agent messages: ~68% of all messages
- Customer messages: ~32% of all messages

## Data Source and License

**TODO**: Document data source and licensing

This dataset is synthetic/anonymized data provided for tutorial purposes.

## Using Your Own Data

To use Agentune Analyze with your own conversation data, structure it similarly:

1. **Main table** - One row per conversation with:
   - Unique conversation identifier
   - Target variable (outcome, label, or metric to predict)
   - Optional: Duration, date, or other conversation-level features

2. **Conversations table** - One row per message with:
   - Conversation identifier (links to metadata)
   - Timestamp (for temporal ordering)
   - Message text content
   - Speaker identifier/role

Both tables can be CSV, Parquet, or loaded directly into DuckDB.

## Questions?

For questions about data format or preparing your own data, see the main documentation or open an issue on GitHub.
