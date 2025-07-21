# Agentune Simulate

Developing your customer-facing conversational AI agent? Want to ensure it behaves as expected before going live? Agentune Simulate is here to help!

Many developers and data scientists struggle to test and validate AI agents effectively. Some deploy directly to production, testing on real customers! Others perform A/B testing, which also means testing on real customers. Many rely on predefined tests that cover main use cases but fail to capture real user intents.

Agentune Simulate creates a customer simulator (twin) based on a set of real conversations. It captures the essence of your customers' inquiries and the way they converse, allowing you to simulate conversations with your AI agent, ensuring it behaves as expected before deployment.

Ready to deploy your improved AI agent? Use Agentune Simulate to validate it first against real customer interactions!

## How It Works

![Agentune Simulate Workflow](https://github.com/SparkBeyond/agentune/blob/main/agentune_simulate/docs/images/agentune-simulate-flow.png)

**How do we validate the twin customer simulator?** We create a twin AI-Agent and let them converse. we then evaluate the conversations to check that the customer simulator behaves as the real customer:

1. **Capture Conversations** - Collect real conversations between customers and your existing AI-agent
2. **Create Simulator** - Create twin Customer Simulator and AI-Agent from the captured conversations
3. **Simulate & Evaluate** - Simulate interactions to evaluate if the twin Customer Simulator behaves as your real customers

**Connect a Real Agent** - Now you can integrate your real agent system and run simulations with simulated customers to validate agent behavior

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

1. **Quick Start** - [`getting_started.ipynb`](https://github.com/SparkBeyond/agentune/blob/main/agentune_simulate/examples/getting_started.ipynb) for a quick getting started example
2. **Production Setup** - [`persistent_storage_example.ipynb`](https://github.com/SparkBeyond/agentune/blob/main/agentune_simulate/examples/persistent_storage_example.ipynb) for a closer to real life, scalable, persistent example  
3. **Validate _Your_ Data** - Adapt the 2nd example to load _your_ conversations data and validate the simulation
4. **Connect Real Agent** - [`real_agent_integration.ipynb`](https://github.com/SparkBeyond/agentune/blob/main/agentune_simulate/examples/real_agent_integration.ipynb) for integrating your existing agent systems

## Contributing

- **Environment Setup**: [Environment Setup Guide](https://github.com/SparkBeyond/agentune/blob/main/agentune_simulate/docs/development/environment-setup.md)
- **Coding Standards**: [Style Guide](https://github.com/SparkBeyond/agentune/blob/main/agentune_simulate/docs/development/style-guide.md)

