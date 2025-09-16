import asyncio
import json
import logging
import re

from llama_index.core.llms import ChatMessage

from agentune.analyze.core.sercontext import LLMWithSpec

logger = logging.getLogger(__name__)


async def achat_raw(llm_with_spec: LLMWithSpec, prompt: str) -> str:
    """Pure I/O wrapper for LLM calls."""
    response = await llm_with_spec.llm.achat([
        ChatMessage(role='user', content=prompt)
    ])
    return response.message.content or ''


async def execute_llm_caching_aware_columnar(llm_with_spec: LLMWithSpec, prompt_columns: list[list[str]]) -> list[list[str]]:
    """Execute LLM calls with caching-aware staging: first column separately, then remaining columns."""
    if not prompt_columns:
        return []
    
    # Stage 1: Execute first column (for prompt cache warming)
    first_column_responses = await asyncio.gather(*[
        achat_raw(llm_with_spec, prompt) for prompt in prompt_columns[0]
    ])
    
    # Stage 2: Execute remaining columns in parallel
    if len(prompt_columns) > 1:
        remaining_responses = await asyncio.gather(*[
            asyncio.gather(*[achat_raw(llm_with_spec, prompt) for prompt in column])
            for column in prompt_columns[1:]
        ])
        return [first_column_responses, *remaining_responses]
    else:
        return [first_column_responses]


def extract_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response."""
    # Look for JSON code blocks using regex
    # Pattern allows for optional newlines after ```json and before ```
    json_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(json_pattern, response, re.DOTALL)
    
    if len(matches) == 0:
        raise ValueError('No JSON found in response')
    elif len(matches) > 1:
        raise ValueError(f'Multiple JSON sections found in response ({len(matches)} sections)')
    
    json_str = matches[0].strip()
    return json.loads(json_str)


def parse_json_response_field(response: str, key: str) -> str | None:
    """Parse response and extract the relevant field."""
    try:
        response_json = extract_json_from_response(response)
        return str(response_json.get(key, '')) if response_json else None
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.warning(f'Failed to parse JSON response field "{key}": {e}')
        return None
