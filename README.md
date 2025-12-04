# Agentune

[![CI](https://github.com/SparkBeyond/agentune/actions/workflows/python-tests.yml/badge.svg?label=CI)](https://github.com/SparkBeyond/agentune/actions)
[![PyPI version](https://badge.fury.io/py/agentune.svg)](https://pypi.org/project/agentune/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Twitter Follow](https://img.shields.io/twitter/follow/agentune_sb?style=social)](https://x.com/agentune_sb)
[![Discord](https://img.shields.io/badge/discord-join-blue?logo=discord&logoColor=white)](https://discord.gg/Hx5YYAaebz)

---

**Open-source framework for continuously improving AI agents.**


Agentune helps teams **analyze, improve, and evaluate** customer-facing AI agents through measurable, data-driven iterations — not guesswork.


Instead of tweaking prompts and hoping for the best, Agentune connects **real conversations**, **context data**, and **simulations** into a repeatable optimization loop that drives actual KPI improvements such as conversion, CSAT, and retention.


---


## Why Agentune


Most agents are launched and left to stagnate — tuned by intuition, not evidence.


Agentune enables **continuous agent improvement** by combining analytics, optimization, and simulation in a single open framework:


- **Analyze** – uncover what drives your agent’s KPIs up or down  
- **Improve** – generate actionable recommendations to lift performance  
- **Simulate** – safely test and benchmark improvements before deployment  


The result: agents that don’t just respond — they **learn what works**.


---


## The agentune-simulate library

[Agentune Simulate](agentune_simulate/README.md) is a separately installable library that enables you to create customer simulations to test and benchmark your agent's behavior before production.

Together with agentune, it forms the **Analyze → Improve → Simulate** loop — a disciplined framework for building smarter, higher-performing AI agents.

A future version of agentune-simulate will merge it into the main agentune package.

---


## Real-World Use Cases


Agentune is built for teams who want to move beyond trial-and-error:


- **AI platform / infra teams** managing production-grade agents across multiple domains or use cases  
- **ML / data teams** accountable for KPI impact, not just model accuracy  
- **Product / ops teams** who need to measure and harden conversational behavior before it reaches users  


Common scenarios:
- Diagnose why conversion or CSAT is dropping  
- Quantify which behaviors, intents, or flows impact KPIs  
- Test new prompt or policy versions safely  
- Continuously improve deployed agents over time  


---

## Agentune Analyze & Improve

**Turn real conversations into insights that measurably improve your AI agents.**


Agentune Analyze & Improve helps teams discover what drives an agent’s KPIs up or down — and generate concrete recommendations to enhance performance.  
It transforms messy operational data into interpretable, data-driven actions that actually move business metrics.


---


### Why It Matters


Most AI agents are optimized by intuition: a few sample chats, some prompt edits, and best guesses.


Agentune replaces guesswork with evidence.  
Using structured and unstructured data from real conversations, it:


- Identifies **patterns** that correlate with KPI outcomes
- Surfaces **interpretable insights** (not opaque scores)
- Recommends **targeted changes** to prompts, policies, and logic


No more trial-and-error tuning — just measurable improvement grounded in data.


For example: suppose you built a sales agent and now have a dataset of conversations with labeled outcomes as **win**, **undecided**, or **lost**.
Using Agentune Analyze & Improve, you can discover insights showing which patterns or intents correlate with those outcomes and receive concrete recommendations to refine the agent’s playbook — for instance, improving how it handles discounts, competitor mentions, or shipping questions.




### How It Works


Agentune Analyze & Improve follows a transparent, two-step process:


#### 1. Analyze
- Ingests conversations, outcomes, and optional context data (e.g., product, policy, CRM).
- Generates semantic and structural **features** that capture patterns in language, behavior, or flow.
- Selects statistically significant features correlated with KPI changes — these become your **drivers** of performance.


Example insights:
- “Mentions of competitors early in chat increase conversion probability.”
- “Discount discussion combined with shipping-time questions lowers CSAT.”


#### 2. Improve
- Maps the discovered drivers into **actionable recommendations** — changes to prompts, tool usage, escalation logic, or playbooks.
- Outputs a ranked list of improvement opportunities, each linked to its supporting data.


These recommendations can then be validated using [Agentune Simulate](https://github.com/SparkBeyond/agentune/blob/main/agentune_simulate/README.md) before deployment.


---


### Example Usage

1. **Getting Started** - [`01_getting_started.ipynb`](https://github.com/SparkBeyond/agentune/blob/main/examples/01_getting_started.ipynb) for an introductory walkthrough of library fundamentals
2. **End-to-End Script Example** - [`e2e_script_example.md`](https://github.com/SparkBeyond/agentune/blob/main/examples/e2e_script_example.md) - a runnable example executing the entire analysis workflow
3. **Advanced Examples** - [`advanced_examples.md`](https://github.com/SparkBeyond/agentune/blob/main/examples/advanced_examples.md) for customizing components, using LLM requests caching, and advanced workflows


### Testing & Costs
We've tested Agentune Analyse with the combination of OpenAI o3 and gpt-4o-mini. In our tests, the cost per conversation was approximately 5-10 cents per conversation.


### Installation


```bash
pip install agentune
```


**Requirements**
- Python ≥ 3.12
- Note for Mac users: If you encounter errors related to lightgbm, you may need to install OpenMP first: brew install libomp. See the LightGBM macOS installation guide for details.


---


### Key Features


- 🧩 **Feature Generation** – semantic, structural, and behavioral signals derived from real interactions
- 📈 **Feature Selection** – statistical and semantic correlation with target KPIs
- 💡 **Actionable Insights** – interpretable drivers with examples and metrics
- 🧠 **Context Awareness (upcoming)** – integrates CRM, product, and policy metadata for deeper understanding


---


## Roadmap


**Current focus:** advancing **Analyze & Improve** with structured, context-aware optimization.


Planned milestones:
- Context-aware **feature generation and insight discovery**  
- Integration of **context features** into the **recommendation layer** for targeted improvement actions  
- Expanded evaluation and visualization tooling for Analyze & Improve results  
- Visualization tools for insight exploration
- Seamless flow into `agentune-simulate` for validating improvements


**Longer-term:**
- Multi KPI analytics: understand how improving one KPI impacts other KPIs and account for that in the suggested improvement recommendations.
- Optional multi-agent analytics and cross-agent benchmarking  


---


## Contributing


We welcome contributions from engineers who care about **robust, measurable agents**.

- Open issues for bugs, integrations, or feature proposals  
- Early adopters: reach us at **agentune-dev@sparkbeyond.com**
- 💬 **Join our community on [Discord](https://discord.gg/Hx5YYAaebz)** to connect with maintainers, share ideas, and get support 


---
