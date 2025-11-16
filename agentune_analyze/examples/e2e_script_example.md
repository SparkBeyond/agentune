# End-to-End Simple Example

A minimal runnable example demonstrating the complete Agentune Analyze workflow: loading conversation data, running analysis, and generating action recommendations.

## Prerequisites

```bash
pip install agentune-analyze
export OPENAI_API_KEY="your-api-key"
```

## Complete Code

```python
import asyncio
import os
from pathlib import Path

from agentune.analyze.api.base import LlmCacheOnDisk, RunContext
from agentune.analyze.feature.problem import ProblemDescription


async def main() -> None:
    data_dir = Path(__file__).parent / 'data'
    
    # Define the problem
    problem = ProblemDescription(
        target_column='outcome',
        problem_type='classification',
        target_desired_outcome='process paused - customer needs to consider the offer',
        name='Customer Service Conversation Outcome Prediction',
        description='Analyze auto insurance conversations and suggest improvements',
        target_description='The final outcome of the conversation'
    )

    # Create run context with LLM caching
    async with await RunContext.create(
        llm_cache=LlmCacheOnDisk(str(Path(__file__).parent / 'llm_cache.db'), 300_000_000)
    ) as ctx:
        # Load data
        conversations_table = await ctx.data.from_csv(
            str(data_dir / 'conversations.csv')
        ).copy_to_table('conversations')
        
        messages_table = await ctx.data.from_csv(
            str(data_dir / 'messages.csv')
        ).copy_to_table('messages')

        # Configure join strategy
        join_strategy = messages_table.join_strategy.conversation(
            name='messages',
            main_table_key_col='conversation_id',
            key_col='conversation_id',
            timestamp_col='timestamp',
            role_col='author',
            content_col='message'
        )
        
        # Split data
        split_data = await conversations_table.split(train_fraction=0.9)

        # Run analysis
        results = await ctx.ops.analyze(
            problem_description=problem,
            main_input=split_data,
            secondary_tables=[messages_table],
            join_strategies=[join_strategy]
        )

        # Generate recommendations
        recommendations = await ctx.ops.recommend_actions(
            analyze_input=split_data,
            analyze_results=results,
            recommender=ctx.defaults.conversation_action_recommender()
        )


if __name__ == '__main__':
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError('Please set OPENAI_API_KEY environment variable')
    asyncio.run(main())
```

## Usage

```bash
cd agentune_analyze/examples
python e2e_simple_example.py
```

The script will analyze conversations, discover patterns correlated with outcomes, and generate actionable recommendations.

## Learn More

- [Getting Started Notebook](https://github.com/SparkBeyond/agentune/blob/main/agentune_analyze/examples/01_getting_started.ipynb) - Interactive walkthrough with detailed explanations
- [Advanced Examples](https://github.com/SparkBeyond/agentune/blob/main/agentune_analyze/examples/advanced_examples.md) - Customization and advanced workflows
- [Data README](https://github.com/SparkBeyond/agentune/blob/main/agentune_analyze/examples/data/README.md) - Data format documentation
