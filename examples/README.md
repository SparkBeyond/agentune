# Examples

This directory contains example scripts demonstrating how to use the conversation simulator library.

## Available Examples

### simple_rag_simulation.py

A complete RAG-based conversation simulation example that uses InMemoryVectorStore with zero external dependencies.

Note that this example is designed to be simple and straightforward, focusing on the core functionality of the library without additional complexities. In real applications, you might want to use more advanced vector stores and index the data once rather than loading it each simulation session.

**Features:**
- Loading reference conversations from JSON data
- Building a single vector store for all RAG components using InMemoryVectorStore
- Using the opinionated SimulationSessionBuilder with RAG-based participants
- Running simulation session to generate 20 conversations
- Analyzing and saving results
- Uses cattrs for structured data loading

**Core Function:**
The main entry point is [`run_rag_simulation()`](./simple_rag_simulation.py#L97-L160) which provides a clean interface for running the complete simulation pipeline.

**Requirements:**
- OpenAI API key set in environment (`OPENAI_API_KEY`)
- No additional vector store dependencies required!

**Usage:**
```bash
# Simple usage with all defaults, execute from the root of the repository:
python examples/simple_rag_simulation.py
```

**Default Behavior:**
- Uses `gpt-4o-mini` model and `text-embedding-3-small` embeddings
- Loads first 20 conversations from `tests/data/dch2_sampled_dataset.json`
- Extracts outcomes automatically from conversation data
- Runs simulation with max 20 messages per conversation
- Saves results to `simple_simulation_results.json`

**What it demonstrates:**
1. **Data Loading**: Parse real conversation data from JSON using cattrs
2. **Outcome Extraction**: Automatically extract unique outcomes from conversations
3. **Vector Store Setup**: Create a single InMemoryVectorStore for all RAG components
4. **Session Building**: Use SimulationSessionBuilder with opinionated RAG/zero-shot defaults
5. **Execution**: Run the complete simulation pipeline
6. **Analysis**: Generate outcome distributions and analysis
7. **Results Export**: Save structured results to JSON

## Code Structure Reference

For developers who want to understand or modify the example:

- **Main execution flow**: [`main()`](./simple_rag_simulation.py#L209-L250)
- **Data loading utilities**: [`load_sample_conversations()`](./simple_rag_simulation.py#L44-L65), [`extract_outcomes_from_conversations()`](./simple_rag_simulation.py#L69-L93)
- **Core simulation logic**: [`run_rag_simulation()`](./simple_rag_simulation.py#L97-L160)
- **Results analysis**: [`print_simple_summary()`](./simple_rag_simulation.py#L161-L207)

### Key Library Components Used

- **Models**: [`Conversation`](../conversation_simulator/models/conversation.py), [`Outcomes`](../conversation_simulator/models/outcome.py), [`SimulationSessionResult`](../conversation_simulator/models/results.py)
- **Session Builder**: [`SimulationSessionBuilder`](../conversation_simulator/simulation/session_builder.py) - Uses opinionated RAG/zero-shot defaults
- **RAG Utilities**: [`conversations_to_langchain_documents`](../conversation_simulator/rag/indexing_and_retrieval.py)

**Note**: The new SimulationSessionBuilder automatically creates RAG-based participants and zero-shot components, eliminating the need for manual factory construction.

#### Example Data

The examples use sample data from `tests/data/` directory:
- `dch2_sampled_dataset.json`: Customer service conversations from the DHC2 dataset

You can also provide your own conversation data in the same JSON format, or load from a different source by modifying the `load_reference_conversations` function:

```json
{
  "comments": "Description of your dataset",
  "conversations": [
    {
      "messages": [
        {
          "sender": "customer",
          "content": "Message content here",
          "timestamp": "2024-01-15T09:00:00Z"
        },
        {
          "sender": "agent", 
          "content": "Agent response here",
          "timestamp": "2024-01-15T09:02:00Z"
        }
      ],
      "outcome": {
        "name": "resolved",
        "description": "Issue was resolved successfully"
      }
    }
  ]
}
```

## Troubleshooting

### OpenAI API Issues
- Verify your API key is set correctly: `echo $OPENAI_API_KEY`

## Contributing

To add new examples:
1. Create a new Python script in this directory
2. Follow the pattern established in `simple_rag_simulation.py`
3. Add documentation to this README
