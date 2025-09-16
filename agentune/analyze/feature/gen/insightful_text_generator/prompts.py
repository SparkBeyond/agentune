"""Prompt templates and utilities for query generation."""

# Main prompt template for query generation
from agentune.analyze.core import types
from agentune.analyze.feature.base import CategoricalFeature
from agentune.analyze.feature.gen.insightful_text_generator.schema import PARSER_OUT_FIELD

QUESTIONNAIRE_PROMPT = '''
Below are a set of {instance_full_description}.

We want to understand what is special about those with {target} = {desired_target_value} and what is special about those with {target} != {desired_target_value}.
For this we want to prepare a questionnaire that we will run per each conversation. Then we will give our data scientists department to analyze the results.

For each {instance} we have the following fields (some may be omitted):
{field_descriptions}.

We will give you sample conversations with various status values and then the requested output format

###### {instance} examples ###
{examples}

###### output format ######
Please prepare a questionnaire of up to {n_queries} questions that can be applied to
{instance}.
We will then give our data scientists to analyze the results in order to better understand
what characterizes the {desired_target_value} cases.

Focus of the Questions:
Focus on technical and process-related aspects of the {instance} (e.g., in sales, product discussed, customer intent, objections raised, assistance steps provided).
Avoid interpersonal or stylistic questions (e.g., tone or politeness).

Aim for short, structured answers such as:
* Yes/No (e.g., Y / N)
* Numbers (e.g., number of products mentioned)
* Predefined categories (e.g., in sales, what is the product category (+ list possible answers)?)
* Short texts (e.g., what is the product name, what is the customer ask)

End Goal:
We will apply these questions automatically to thousands of {instance_full_description}.
The structured output will be used by the Data Science team to analyze and improve assistant behavior.


Please work step by step.
First explain your thoughts, then return the result in the format below.
Please use the format
{{
    <informative question name>: <question text>
}}
Please respond with a proper json. Your answer will be parsed by a computer, so please ensure it is well-structured and valid JSON.
The json code block should start with three backticks and end with three backticks, like this:
```json
{{
    "question_1": "What is the product category?",
    "question_2": "How many products were discussed?",
    ...
}}
```
'''


def create_questionnaire_prompt(
    examples: str,
    instance: str,
    instance_full_description: str,
    target: str,
    field_descriptions: str,
    desired_target_value: str,
    n_queries: str
) -> str:
    """Create a formatted questionnaire prompt for LLM analysis.
    
    Args:
        examples: Formatted sample data
        instance: What each data point represents (e.g., "conversation")
        instance_full_description: Full description (e.g., "conversations between customer and agent")
        target: Target variable name to analyze
        field_descriptions: Description of available data fields
        desired_target_value: The target value we want to characterize
        n_queries: Number of queries to generate
        
    Returns:
        Formatted prompt string ready for LLM
    """
    return QUESTIONNAIRE_PROMPT.format(
        instance_full_description=instance_full_description,
        instance=instance,
        target=target,
        field_descriptions=field_descriptions,
        desired_target_value=desired_target_value,
        n_queries=n_queries,
        examples=examples
    )


QUERIES_ENHANCEMENT_PROMPT = f'''
Below is a single instance from a dataset.
It contains the following fields (some may be missing):
{{instance_description}}
Below is the instance followed by a task to perform on it.

Instance
##########

{{instance}}

Task
##########

As data scientists, we are interested in the following high-level question:
{{queries_str}}

For the data instance above, please provide an answer to the question.
Your format should be a dictionary with a single "{PARSER_OUT_FIELD}" key containing your answer.
Please respond with a proper json. Your answer will be parsed by a computer, so please ensure it is well-structured and valid JSON.
The json code block should start with three backticks and end with three backticks, like this:
```json
{{{{
    "{PARSER_OUT_FIELD}": <answer>
}}}}
```
'''


def create_enrich_conversation_prompt(
    instance: str,
    queries_str: str,
    instance_description: str
) -> str:
    """Create the enrich conversation prompt using f-string formatting.
    
    Args:
        instance: Formatted instance of data row
        queries_str: String representation of queries to answer
        instance_description: Description of the instances
        
    Returns:
        Formatted prompt string ready for LLM
    """
    return QUERIES_ENHANCEMENT_PROMPT.format(
        instance=instance,
        queries_str=queries_str,
        instance_description=instance_description
    )


CATEGORICAL_OPTIMIZER_PROMPT = '''You are a data analysis expert. Your task: optimize categorical features for better consistency and usability.

**Goal**: Analyze a feature query and its historical answers to create an improved query with well-defined categories.

**Input:**
- Query Name: {query_name}
- Original Query: {query_text}
- Max Categories: {max_categorical} (not including 'Others')
- Historical Answers Histogram: {answers_hist}

**Objectives**: Create a refined query and categories that:
1. Maintain the original intent
2. Produce exactly ONE answer per input (not multiple)
3. Group similar historical answers into consistent categories
4. Use at most {max_categorical} distinct, informative categories
5. Cover most common expected answers (rare cases will be classified as "Others" by the model)
6. Do not return 'Others' category, it will be added automatically after

**Key Requirements:**
- Frame questions for single answers (ask for "main" or "primary" rather than "all")
- Categories should be distinct and cover the most frequent/important answers
- Use the histogram to understand which answers are most common and should get their own categories
- Keep category names clear and unambiguous
- Preserve the core meaning of the original feature

**Examples:**
- "USA", "United States", "US", "America" → Category: "United States"
- "very positive", "extremely positive", "positive" → Category: "Positive"
- "quick", "fast", "rapid", "speedy" → Category: "Fast"

**Output:**
- query_name: Clear feature name
- categories: List of category names (≤ {max_categorical})
- query_text: Refined query that maps to your categories
'''


QUERY_FEATURE_PROMPT = '''
Below is a single instance from a dataset.
It contains the following fields (some may be missing):
{instance_description}
Below is the instance followed by a task to perform on it.

Instance
##########

{instance}

Task
##########

As data scientists, we are interested in the following high-level question/feature:
{query_text}

For the data instance above, please provide an answer for this question.

{output_instructions}
'''


def get_output_instructions(dtype: types.Dtype) -> str:
    """Generate output instructions based on the data type."""
    if dtype == types.boolean:
        return f'''Output Instructions:
Your answer should be a boolean value (true or false).
Example: {{"{PARSER_OUT_FIELD}": true}}'''

    elif dtype == types.int32:
        return f'''Output Instructions:
Your answer should be an integer value.
Example: {{"{PARSER_OUT_FIELD}": 42}}'''

    elif dtype == types.float64:
        return f'''Output Instructions:
Your answer should be a numeric value (integer or float).
Example: {{"{PARSER_OUT_FIELD}": 3.14}}'''

    elif isinstance(dtype, types.EnumDtype):
        categories = list(dtype.values)
        categories_str = ', '.join(f'"{cat}"' for cat in categories)
        return f'''Output Instructions:
Your answer should be one of the following categories: {categories_str}.
If the answer doesn't fit any of the specific categories, use "{CategoricalFeature.other_category}".
Example: {{"{PARSER_OUT_FIELD}": "{categories[0]}"}}'''

    raise NotImplementedError(f'Output instructions not implemented for dtype: {dtype}')
