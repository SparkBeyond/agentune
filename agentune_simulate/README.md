# Agentune Simulate

Developing your customer-facing conversational AI agent? Want to ensure it behaves as expected before going live? Agentune Simulate is here to help!

Many developers and data scientists struggle to test and validate AI agents effectively. Some deploy directly to production, testing on real customers! Others perform A/B testing, which also means testing on real customers. Many rely on predefined tests that cover main use cases but fail to capture real user intents.

Agentune Simulate creates a customer simulator (twin) based on a set of real conversations. It captures the essence of your customers' inquiries and the way they converse, allowing you to simulate conversations with your AI agent, ensuring it behaves as expected before deployment.

Ready to deploy your improved AI agent? Use Agentune Simulate to validate it first against real customer interactions!

## How It Works

![Agentune Simulate Workflow](https://github.com/SparkBeyond/agentune/blob/main/agentune_simulate/docs/images/agentune-simulate-flow.png)

Agentune Simulate follows a three-step process:

1. **Capture Conversations** - Collect real conversations between customers and your existing AI agent
2. **Create Simulator** - Create a Twin Customer Simulator from the captured conversations
3. **Simulate & Evaluate** - Simulate interactions to evaluate if your improved AI agent behaves as expected

## Quick Start

### Install Agentune Simulate

   ```bash
   pip install agentune-simulate
   ```

### Basic usage example

   ```python
   from agentune.simulate import SimulationSessionBuilder
   from langchain_openai import ChatOpenAI
   
   # Load your conversations and create outcomes
   session = SimulationSessionBuilder(
       default_chat_model=ChatOpenAI(model="gpt-4o"),
       outcomes=outcomes,
       vector_store=vector_store
   ).build()
   
   # Run simulation
   results = await session.run_simulation(real_conversations=conversations)
   ```

### Learn with examples

Start with [`getting_started.ipynb`](https://github.com/SparkBeyond/agentune/blob/main/agentune_simulate/examples/getting_started.ipynb) for a complete tutorial  
See [`persistent_storage_example.ipynb`](https://github.com/SparkBeyond/agentune/blob/main/agentune_simulate/examples/persistent_storage_example.ipynb) for Chroma vector store and scaling

## Contributing

- **Environment Setup**: [https://github.com/SparkBeyond/agentune/blob/main/agentune_simulate/docs/development/environment-setup.md](./docs/development/environment-setup.md)
- **Coding Standards**: [https://github.com/SparkBeyond/agentune/blob/main/agentune_simulate/docs/development/style-guide.md](./docs/development/style-guide.md)

