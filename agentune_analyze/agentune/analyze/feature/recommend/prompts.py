"""Prompts for feature recommendation."""

from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel

from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.feature.problem import Problem, RegressionDirection, RegressionProblem

# Default descriptions for the agent and instances being analyzed
DEFAULT_AGENT_DESCRIPTION = 'An advanced online sales agent designed to assist customers with product inquiries and purchases.'


# Prompt template adapted from recommendations_report/prompt.py
CONVERSATION_ANALYSIS_PROMPT = '''
We have an agent that we want to improve.
{agent_description}

We did an extensive semantic analysis of {instance_description} 
{comparison_description}

We will give you 
(1) the results of this analysis - features along with their predictive power measured by SSE reduction
(2) Sample conversations of the agent with different outcomes 
Please observe both carefully. We will then ask you to create a prioritized implementation plan for the agent builder.
(3) Additional guidelines and the output format we expect

IMPORTANT: When analyzing conversations, pay special attention to cases where the desired outcome was NOT achieved.
Identify specific moments or missing actions in those conversations that could have changed the outcome.
These negative examples are extremely valuable for understanding what to avoid or what to add.

---
Features found during our Semantic analysis along with their SSE reductions (higher = more predictive)

{sse_reduction_dict}

---
Sample Conversations with various outcomes
{conversations}

---
Task definition:

Please observe the above carefully. Is there anything important to recommend to the agent builders?  
If yes, please write it. 
Do not just describe *what* to do; explain *how* to do it.

For each recommendation, include:

**Finding (The "What"):**
State the user-facing problem or business-level issue concisely.

**Analysis & Impact (The "Why"):**
Describe the root cause and its impact. Support your analysis with:
- Quantitative evidence (SSE scores, percentages, patterns in the data)
- Qualitative evidence (specific examples from conversations)
- Business impact (effect on sales, user experience, etc.)

**Strategic Recommendation (The "What Next?"):**
Provide a product-level recommendation. Explain what feature, flow, or capability should be prioritized and how to approach it. Be specific and actionable.

---
{goal_description}
'''


def create_conversation_analysis_prompt(
    agent_description: str,
    instance_description: str,
    problem: Problem,
    sse_reduction_dict: str,
    conversations: str,
) -> str:
    """Create a formatted conversation analysis prompt for LLM.
    
    This function adapts the prompt based on whether the problem is regression or classification,
    matching the pattern from create_questionnaire_prompt in insightful_text_generator.
    
    Args:
        agent_description: Description of the agent being analyzed
        instance_description: Description of the instances (e.g., "transcripts of conversations")
        problem: Problem object containing target and desired outcome
        sse_reduction_dict: Formatted string of features and their SSE reductions
        conversations: Formatted string of sample conversations
        
    Returns:
        Formatted prompt string ready for LLM
    """
    target_name = problem.target_column.name
    
    if isinstance(problem, RegressionProblem):
        # For regression problems, target_desired_outcome is RegressionDirection (up/down)
        direction = 'high' if problem.target_desired_outcome == RegressionDirection.up else 'low'
        other_direction = 'low' if direction == 'high' else 'high'
        
        comparison_description = (
            f'to understand what distinguishes cases with {direction} {target_name} values '
            f'from cases with {other_direction} {target_name} values.'
        )
        goal_description = (
            f'Focus your analysis on understanding what leads to {direction}er {target_name} values. '
        )
    else:
        # For classification problems, target_desired_outcome is a specific class value
        desired_value = problem.problem_description.target_desired_outcome
        comparison_description = (
            f'to understand what is special about those with {target_name} = {desired_value} '
            f'and what is special about those with {target_name} != {desired_value}.'
        )
        goal_description = (
            f'Focus your analysis on understanding what characterizes the {desired_value} cases. '
        )
    
    return CONVERSATION_ANALYSIS_PROMPT.format(
        agent_description=agent_description,
        instance_description=instance_description,
        comparison_description=comparison_description,
        goal_description=goal_description,
        sse_reduction_dict=sse_reduction_dict,
        conversations=conversations,
    )


# Structured output classes (Pydantic models for LLM structured output)
class ConversationReference(BaseModel):
    """Reference to a conversation with explanation of relevance."""
    conversation_id: int
    explanation: str

class RecommendationRaw(BaseModel):
    """An actionable recommendation."""
    title: str
    description: str
    rationale: str
    evidence: str
    supporting_features: list[str]
    supporting_conversations: list[ConversationReference]


class StructuredReport(BaseModel):
    """Structured JSON format of the conversation recommendation report.
    
    This is the target schema for converting text reports to JSON (Pydantic for LLM).
    """
    analysis_summary: str
    recommendations: list[RecommendationRaw]


# Prompt for structuring text report to JSON
STRUCTURING_PROMPT_TEMPLATE = '''You are a precise, inferential information extraction system. Your task is to extract and structure information from the report below WITHOUT rephrasing, summarizing, or adding any interpretation.

The report may follow a flexible structure, such as providing an "Observation" (the problem or rationale) followed by a "→ Recommendation" (the description or action). You must map this pattern to the requested fields.

---
REPORT:
{report}
---

AVAILABLE FEATURES (for exact matching):
{sse_reduction_dict}
---

Extract information EXACTLY as written in the report. Do not rephrase or rewrite.

**CRITICAL: Preserve formatting and readability:**
- When extracting multi-paragraph text, preserve paragraph breaks by including newline characters (\n\n between paragraphs)
- For bullet points or lists, include \n between items
- For line breaks within a section, use \n
- **Preserve indentation**: Keep leading spaces/tabs for nested lists (e.g., "   a. Sub-item" should maintain the spaces before "a.")
- This ensures the extracted text remains readable and properly structured when displayed

For each distinct recommendation you can identify (even if not in a "# Recommendations" section):

1.  **Extract the title:**
    * Use the numbered heading (e.g., "1. What separates 'wins' from 'losses'").
    * If no clear heading exists for a recommendation, create a concise title based on the description and rationale.

2.  **Extract the rationale:**
    * Extract the content that explains why this recommendation matters (the problem, root cause, and impact).
    * Extract only the content itself, not section headers or labels.

3.  **Extract the description:**
    * Extract the recommended solution - the action plan explaining what to do and how to approach it.
    * Include technical details, specifications, and implementation guidance.
    * Extract only the content itself, not section headers or labels.

4.  **Extract the evidence:**
    * Extract specific examples or observations that support this recommendation.
    * Extract only the content itself, not section headers or labels.

5.  **Extract supporting_features:**
    * Look for mentions of features in the recommendation text
    * For each feature mentioned, find and return its exact description from the "AVAILABLE FEATURES" list above
    * Return ONLY the feature description text (without the list number or SSE score)
    * Do NOT paraphrase or reword - copy the text exactly as it appears in the list

6.  **Extract supporting_conversations:**
    * Find all conversation numbers mentioned. Be flexible: look for "Conversation #" *and* "Example #" (e.g., "Example #12" or "Conversation 15").
    * For each conversation, extract the explanation of why it's relevant.
    * Example: "Example #12 (lost): customer asks about package size..." → conversation_id=12, explanation="customer asks about package size; agent: \"Unfortunately I don't have…\" (no workaround offered)."
    * Include both positive examples (what worked) and negative examples (what didn't work).

Do NOT create new explanations. Extract them directly from the report text.
'''


async def structure_report_with_llm(
    report: str,
    sse_reduction_dict: str,
    model: LLMWithSpec,
    structuring_model: LLMWithSpec | None = None,
) -> StructuredReport:
    """Convert a text report to structured format using LLM structured output.
    
    Args:
        report: The full text report (with # Analysis and # Recommendations sections)
        sse_reduction_dict: Formatted string of features and their SSE reductions
        model: LLM model to use for structuring (fallback if structuring_model not provided)
        structuring_model: Optional faster model specifically for structuring (e.g., gpt-4o)
        
    Returns:
        StructuredReport (Pydantic model from LLM)
    """
    prompt = PromptTemplate(STRUCTURING_PROMPT_TEMPLATE)

    # Use structuring_model if provided, otherwise fall back to main model
    llm_to_use = structuring_model.llm if structuring_model else model.llm
    
    return await llm_to_use.astructured_predict(
        output_cls=StructuredReport,
        prompt=prompt,
        report=report,
        sse_reduction_dict=sse_reduction_dict,
    )
