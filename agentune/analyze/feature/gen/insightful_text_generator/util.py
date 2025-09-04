import asyncio
import json
import logging

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
    def _raise_no_json_error() -> None:
        raise ValueError('No JSON found in response')
    
    start = response.find('{')
    end = response.rfind('}') + 1
    if start != -1 and end > start:
        json_str = response[start:end]
    else:
        _raise_no_json_error()
    
    return json.loads(json_str)


def parse_json_response_field(response: str, key: str) -> str | None:
    """Parse response and extract the relevant field."""
    try:
        response_json = extract_json_from_response(response)
        return str(response_json.get(key, '')) if response_json else None
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.warning(f'Failed to parse JSON response field "{key}": {e}')
        return None
