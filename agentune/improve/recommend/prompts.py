"""Prompts for feature recommendation."""

from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel

from agentune.analyze.feature.problem import Problem, RegressionDirection, RegressionProblem
from agentune.core.sercontext import LLMWithSpec

# Default descriptions for the agent and instances being analyzed
DEFAULT_AGENT_DESCRIPTION = 'An advanced online sales agent designed to assist customers with product inquiries and purchases.'


# Prompt template adapted from recommendations_report/prompt.py
CONVERSATION_ANALYSIS_PROMPT = '''
We have an agent that we want to improve.
{agent_description}

We did an extensive semantic analysis of {instance_description} 
{comparison_description}

We will give you 
(1) the results of this analysis - features along with their predictive power measured by R squared
(2) Sample conversations of the agent with different outcomes 
Please observe both carefully. We will then ask you to create a prioritized implementation plan for the agent builder.
(3) Additional guidelines and the output format we expect

IMPORTANT: When analyzing conversations, pay special attention to cases where the desired outcome was NOT achieved.
Identify specific moments or missing actions in those conversations that could have changed the outcome.
These negative examples are extremely valuable for understanding what to avoid or what to add.

---
Features found during our Semantic analysis along with their R squared values (higher = more predictive)

{r_squared_dict}

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
- Quantitative evidence (R squared values, percentages, patterns in the data)
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
    r_squared_dict: str,
    conversations: str,
) -> str:
    """Create a formatted conversation analysis prompt for LLM.
    
    This function adapts the prompt based on whether the problem is regression or classification,
    matching the pattern from create_questionnaire_prompt in insightful_text_generator.
    
    Args:
        agent_description: Description of the agent being analyzed
        instance_description: Description of the instances (e.g., "transcripts of conversations")
        problem: Problem object containing target and desired outcome
        r_squared_dict: Formatted string of features and their R squared values
        conversations: Formatted string of sampled conversations
        
    Returns:
        Formatted prompt string ready for LLM
    """
    target_name = problem.target_column.name
    
    if isinstance(problem, RegressionProblem):
        # For regression problems, target_desired_outcome_value is RegressionDirection (up/down)
        direction = 'high' if problem.target_desired_outcome_value == RegressionDirection.up else 'low'
        other_direction = 'low' if direction == 'high' else 'high'
        
        comparison_description = (
            f'to understand what distinguishes cases with {direction} {target_name} values '
            f'from cases with {other_direction} {target_name} values.'
        )
        goal_description = (
            f'Focus your analysis on understanding what leads to {direction}er {target_name} values. '
        )
    else:
        # For classification problems, target_desired_outcome_value is a specific class value
        desired_value = problem.problem_description.target_desired_outcome_value
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
        r_squared_dict=r_squared_dict,
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

class ConversationVerification(BaseModel):
    """Verification result for a single conversation against a recommendation."""
    conversation_id: int  # The conversation number being verified
    supports_recommendation: bool  # True if conversation supports the recommendation


class CleanedEvidenceAndRationale(BaseModel):
    """Evidence and rationale with unverified conversation references removed."""
    cleaned_evidence: str
    cleaned_rationale: str


# Prompt for structuring text report to JSON
STRUCTURING_PROMPT_TEMPLATE = '''You are a precise, inferential information extraction system. Your task is to extract and structure information from the report below WITHOUT rephrasing, summarizing, or adding any interpretation.

The report may follow a flexible structure, such as providing an "Observation" (the problem or rationale) followed by a "→ Recommendation" (the description or action). You must map this pattern to the requested fields.

---
REPORT:
{report}
---

AVAILABLE FEATURES (for exact matching):
{r_squared_dict}
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
    * Return ONLY the feature description text (without the list number or R squared value)
    * Do NOT paraphrase or reword - copy the text exactly as it appears in the list

6.  **Extract supporting_conversations:**
    * Find all conversation numbers mentioned. Be flexible: look for "Conversation #" *and* "Example #" (e.g., "Example #12" or "Conversation 15").
    * For each conversation, extract the explanation of why it's relevant.
    * Example: "Example #12 (lost): customer asks about package size..." → conversation_id=12, explanation="customer asks about package size; agent: \"Unfortunately I don't have…\" (no workaround offered)."
    * Include both positive examples (what worked) and negative examples (what didn't work).

Do NOT create new explanations. Extract them directly from the report text.
'''

# Prompt for verifying a single conversation against a recommendation
CONVERSATION_VERIFICATION_PROMPT_TEMPLATE = '''
CONTEXT:
You are part of an AI agent improvement system. This system analyzes conversations between an AI agent and users to identify patterns that correlate with success or failure.

WHAT IS A RECOMMENDATION:
A recommendation is an actionable suggestion for improving the agent, generated from:
- Semantic analysis showing which conversation features predict outcomes (measured by R squared)
- Sample conversations demonstrating these patterns
- LLM analysis identifying what agent builders should implement

WHAT IS A CONVERSATION:
A conversation is a transcript of messages exchanged between the agent and a user, with a known outcome (e.g., "won" vs "lost", high vs low ratings).

YOUR TASK:
The LLM that generated recommendations cited specific conversations as evidence. However, LLMs can hallucinate or misremember details. Your job is to verify each citation by checking if the conversation actually demonstrates what was claimed.

A conversation "supports" a recommendation when it contains concrete evidence of the specific patterns, behaviors, or issues described in the recommendation.

---

RECOMMENDATION:

Title: {title}

Description: {description}

Rationale: {rationale}

Evidence: {evidence}

CONVERSATION TO VERIFY:

Conversation ID: 
{conversation_id}

Conversation content: 
{conversation}

CONVERSATION OUTCOME: {outcome}

LLM'S CLAIM about why this conversation is relevant:
{explanation}

CRITICAL GUIDELINES:
1. The explanation provides context for why this conversation was cited. Use it as a guide to understand the relevance, but make your own judgment about whether the conversation genuinely supports the recommendation.
2. Outcome as Ground Truth:
The CONVERSATION OUTCOME field above is ground truth. If the explanation claims a different outcome (e.g., explanation says "won" but outcome shows "lost"), trust the actual outcome.

YOUR PRIMARY FOCUS:
Verify whether this conversation demonstrates the pattern, behavior, or issue
described in the recommendation. The explanation is only a guide - it may contain
factual errors (like incorrect outcome classifications/wrong conversation references). Always trust the
CONVERSATION OUTCOME field over any outcome mentioned in the explanation.
If the explanation contains errors but the conversation still demonstrates the pattern described in the recommendation, return True.

Consider:
- Given the actual outcome, does this conversation support the recommendation?
- Does the conversation contain the behaviors/patterns claimed?
- Is there concrete evidence for the specific point being made?

return True if you think the conversation supports the recommendation, False otherwise. 
'''

CONVERSATION_VERIFICATION_PROMPT = PromptTemplate(CONVERSATION_VERIFICATION_PROMPT_TEMPLATE)

# Prompt for cleaning both evidence and rationale together
EVIDENCE_AND_RATIONALE_CLEANING_PROMPT_TEMPLATE = '''Your task is to clean both evidence and rationale text by removing ONLY specific unverified conversation references.

EVIDENCE TEXT:
{evidence}

RATIONALE TEXT:
{rationale}

CONVERSATIONS TO REMOVE (unverified conversations):
{unverified_conversation_ids}

**PRIMARY TASK: Preserve ALL conversation references EXCEPT those explicitly listed above.**

**DEFAULT ACTION: KEEP the conversation reference**
- If a conversation ID is NOT in the "CONVERSATIONS TO REMOVE" list → KEEP IT
- Only remove conversation IDs that explicitly appear in the list above
- Removing a verified conversation (not in the list) is a CRITICAL ERROR

**Step-by-step process:**
1. Find each conversation reference in both the evidence and the rationale texts (e.g., "Conv 22", "Conversation 10", "Convs 2 & 9")
2. For each conversation ID found, check: Is this ID in the CONVERSATIONS TO REMOVE list?
3. Decision logic:
   - If YES (ID is in removal list) → Remove this reference
   - If NO (ID is NOT in removal list) → KEEP this reference (do not modify)
4. After removal, fix grammar (commas, conjunctions)

**Multiple conversations in one sentence:**
- "Convs 2 & 9" where only Conv 2 is in removal list:
  - Remove Conv 2, KEEP Conv 9 → Result: "Conv 9"
- "Convs 2 & 9" where BOTH are in removal list:
  - Remove entire sentence (no evidence remains)

**Single conversation in sentence:**
- "Conversation 22 lasted >45 messages..." where Conv 22 is NOT in removal list:
  - KEEP the entire sentence (Conv 22 is verified)
- "Conversation 10 shows..." where Conv 10 IS in removal list:
  - Remove entire sentence (no supporting evidence)

**Feature references (NEVER remove):**
- DO NOT remove: "Feature #3", "feature 1", "Feature 20", "1. Feature description: 0.1234"
- These are NOT conversation references

Return both the evidence and the rationale cleaned texts.
'''

EVIDENCE_AND_RATIONALE_CLEANING_PROMPT = PromptTemplate(EVIDENCE_AND_RATIONALE_CLEANING_PROMPT_TEMPLATE)

async def structure_report_with_llm(
    report: str,
    r_squared_dict: str,
    model: LLMWithSpec,
    structuring_model: LLMWithSpec | None = None,
) -> StructuredReport:
    """Convert a text report to structured format using LLM structured output.
    
    Args:
        report: The full text report (with # Analysis and # Recommendations sections)
        r_squared_dict: Formatted string of features and their R squared values
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
        r_squared_dict=r_squared_dict,
    )

async def clean_evidence_and_rationale_with_llm(
    evidence: str,
    rationale: str,
    unverified_conversation_ids: set[int],
    model: LLMWithSpec,
    structuring_model: LLMWithSpec | None = None,
) -> tuple[str, str]:
    """Remove unverified conversation references from both evidence and rationale using LLM.

    Args:
        evidence: The evidence text that may contain conversation references
        rationale: The rationale text that may contain conversation references
        unverified_conversation_ids: Set of conversation IDs that were filtered out
        model: LLM model to use for cleaning (fallback if structuring_model not provided)
        structuring_model: Optional faster model specifically for cleaning (e.g., gpt-4o)

    Returns:
        Tuple of (cleaned_evidence, cleaned_rationale) with unverified conversation references removed
    """
    if not unverified_conversation_ids:
        return evidence, rationale

    # Use structuring_model if provided, otherwise fall back to main model
    llm_to_use = structuring_model.llm if structuring_model else model.llm

    # Format conversation IDs as a readable list
    ids_str = ', '.join(f'#{id}' for id in sorted(unverified_conversation_ids))

    result = await llm_to_use.astructured_predict(
        output_cls=CleanedEvidenceAndRationale,
        prompt=EVIDENCE_AND_RATIONALE_CLEANING_PROMPT,
        evidence=evidence,
        rationale=rationale,
        unverified_conversation_ids=ids_str,
    )

    return result.cleaned_evidence, result.cleaned_rationale
