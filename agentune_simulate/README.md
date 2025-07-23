# Agentune Simulate

[![CI](https://github.com/SparkBeyond/agentune/actions/workflows/python-tests.yml/badge.svg?label=CI)](https://github.com/SparkBeyond/agentune/actions)
[![PyPI version](https://badge.fury.io/py/agentune-simulate.svg)](https://pypi.org/project/agentune-simulate/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Twitter Follow](https://img.shields.io/twitter/follow/agentune_sb?style=social)](https://x.com/agentune_sb)
[![Discord](https://img.shields.io/discord/1375004885845807114?color=7289da&label=discord&logo=discord&logoColor=white)](https://discord.gg/Hx5YYAaebz)

---

**Launching an AI Agent? Stop guessing, start simulating.**

Many developers and data scientists struggle to test and validate AI agents effectively. Some deploy directly to production, testing on real customers! Others perform A/B testing, which also means testing on real customers. Many rely on predefined tests that cover main use cases but fail to capture real user intents.

Agentune Simulate creates a customer simulator (twin) based on a set of real conversations. It captures the essence of your customers' inquiries and the way they converse, allowing you to simulate conversations with your AI agent, ensuring it behaves as expected before deployment.

Ready to deploy your improved AI agent? Use Agentune Simulate to validate it first against real customer interactions!

**Need help?** Please contact us. We are committed to assist early adopters in making the most of it!

## How Does It Work?    

Running a simulation with Agentune Simulate generates realistic conversations between your AI agent and simulated customers. This lets you evaluate your agent's performance, identify edge cases, and validate behavior before real deployment.

![Agentune Simulate Workflow](https://raw.githubusercontent.com/SparkBeyond/agentune/main/agentune_simulate/docs/images/agentune-simulate-flow.png)

**How do we validate the twin customer simulator?** We create a twin AI-Agent and let them converse. we then evaluate the conversations to check that the customer simulator behaves as the real customer:

1. **Capture Conversations** - Collect real conversations between customers and your existing AI-agent
2. **Create Simulator** - Create twin Customer Simulator and AI-Agent from the captured conversations
3. **Simulate & Evaluate** - Simulate interactions to evaluate if the twin Customer Simulator behaves as your real customers

![Agentune Simulate Workflow](https://raw.githubusercontent.com/SparkBeyond/agentune/main/agentune_simulate/docs/images/agentune-simulate-validation-flow.png)

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
3. **Validate _Your_ Data** - Adapt the 2nd example to load _your_ conversations data and validate the simulation. 
Here is an example of how to load conversations from tabular data: [`load_conversations_from_csv.ipynb`](https://github.com/SparkBeyond/agentune/blob/main/agentune_simulate/examples/load_conversations_from_csv.ipynb)
4. **Connect Real Agent** - [`real_agent_integration.ipynb`](https://github.com/SparkBeyond/agentune/blob/main/agentune_simulate/examples/real_agent_integration.ipynb) for integrating your existing agent systems

📧 **Need help? Have feedback?** Contact us at [agentune-dev@sparkbeyond.com](mailto:agentune-dev@sparkbeyond.com)

## Contributing

- **Environment Setup**: [Environment Setup Guide](https://github.com/SparkBeyond/agentune/blob/main/agentune_simulate/docs/development/environment-setup.md)
- **Coding Standards**: [Style Guide](https://github.com/SparkBeyond/agentune/blob/main/agentune_simulate/docs/development/style-guide.md)

