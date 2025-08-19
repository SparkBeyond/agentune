"""Prompt templates and utilities for query generation."""

# Main prompt template for query generation
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
