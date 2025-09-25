"""Prompt templates and utilities for query generation."""

# Main prompt template for query generation
from agentune.analyze.core import types
from agentune.analyze.feature.base import CategoricalFeature
from agentune.analyze.feature.gen.insightful_text_generator.schema import PARSER_OUT_FIELD
from agentune.analyze.feature.problem import Problem, RegressionDirection, RegressionProblem

QUESTIONNAIRE_PROMPT = '''
{business_domain_expertise}{problem_context}
Below are a set of {instance_description}.

{comparison_description}
For this we want to prepare a questionnaire that we will run per each {instance_type}. Then we will give our data scientists department to analyze the results.

We will give you sample {instance_type} examples with various {target} values and then the requested output format

###### {instance_type} examples ###
{examples}

###### output format ######
Please prepare a questionnaire of up to {n_queries} questions that can be applied to {instance_type}.
{goal_description}

Focus of the Questions:
Focus on technical and process-related aspects of the {instance_type} (e.g., in sales, product discussed, customer intent, objections raised, assistance steps provided).
Avoid interpersonal or stylistic questions (e.g., tone or politeness).

Aim for short, structured answers such as:
* Yes/No (e.g., Y / N)
* Numbers (e.g., number of products mentioned)
* Predefined categories (e.g., in sales, what is the product category (+ list possible answers)?)
* Short texts (e.g., what is the product name, what is the customer ask)

End Goal:
We will apply these questions automatically to thousands of {instance_description}.
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
    problem: Problem,
    instance_type: str,
    instance_description: str,
    n_queries: str
) -> str:
    """Create a formatted questionnaire prompt for LLM analysis.
    
    Args:
        examples: Formatted sample data
        problem: Problem object containing target, business domain, and descriptions
        instance_type: Type of the instance being analyzed (e.g. "conversation")
        instance_description: Description of the instance being analyzed (recommended: DataFormatter.description)
        n_queries: Number of queries to generate
        
    Returns:
        Formatted prompt string ready for LLM
    """
    # Build optional context sections
    business_domain_expertise = ''
    if problem.problem_description.business_domain:
        business_domain_expertise = f'You are an expert in {problem.problem_description.business_domain}.\n\n'
    
    problem_context = ''
    context_parts = []
    
    if problem.problem_description.name:
        context_parts.append(f'Problem: {problem.problem_description.name}')
    
    if problem.problem_description.description:
        context_parts.append(f'Description: {problem.problem_description.description}')
    
    if context_parts:
        problem_context = '\n'.join(context_parts) + '\n'

    if not problem.problem_description.target_desired_outcome:
        raise ValueError('Problem description must include target_desired_outcome.')

    # Create different descriptions for regression vs classification
    target_name = problem.target_column.name
    
    if isinstance(problem, RegressionProblem):
        # For regression problems, target_desired_outcome is RegressionDirection (up/down)
        direction = 'high' if problem.target_desired_outcome == RegressionDirection.up else 'low'
        other_direction = 'low' if direction == 'high' else 'high'
        
        comparison_description = f'We want to understand what distinguishes cases with {direction} {target_name} values from cases with {other_direction} {target_name} values.'
        goal_description = f'We will then give our data scientists to analyze the results in order to better understand what leads to {direction}er {target_name}.'
    else:
        # For classification problems, target_desired_outcome is a specific class value
        desired_value = problem.problem_description.target_desired_outcome
        comparison_description = (f'We want to understand what is special about those with {target_name} = {desired_value} '
                                  f'and what is special about those with {target_name} != {desired_value}.')
        goal_description = f'We will then give our data scientists to analyze the results in order to better understand what characterizes the {desired_value} cases.'

    return QUESTIONNAIRE_PROMPT.format(
        business_domain_expertise=business_domain_expertise,
        problem_context=problem_context,
        instance_type=instance_type,
        instance_description=instance_description,
        target=target_name,
        comparison_description=comparison_description,
        goal_description=goal_description,
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
