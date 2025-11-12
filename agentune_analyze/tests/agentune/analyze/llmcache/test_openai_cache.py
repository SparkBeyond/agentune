import logging
from collections.abc import MutableMapping
from typing import Any

import httpx
import pytest
from frozendict import frozendict
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
    TextBlock,
)
from llama_index.llms.openai import OpenAI, OpenAIResponses

from agentune.analyze.llmcache.base import CanonicalizedChatMessage, LLMCacheBackend, LLMCacheKey
from agentune.analyze.llmcache.openai_cache import CachingOpenAI, CachingOpenAIResponses

_logger = logging.getLogger(__name__)


@pytest.fixture
def openai_llm(httpx_async_client: httpx.AsyncClient) -> OpenAI:
    """For now, relies on local env var containing credentials. Tests using this can't run on the CI yet."""
    return OpenAI('gpt-4.1-mini', temperature=0.0, async_http_client=httpx_async_client)

@pytest.fixture
def openai_responses_llm(httpx_async_client: httpx.AsyncClient) -> OpenAIResponses:
    """For now, relies on local env var containing credentials. Tests using this can't run on the CI yet."""
    return OpenAIResponses('gpt-4.1-mini', temperature=0.0, async_http_client=httpx_async_client)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_cache(openai_llm: OpenAI) -> None:
    cache: MutableMapping[Any, Any] = {}
    caching_openai = CachingOpenAI.adapt(openai_llm, cache)

    assert caching_openai._cache is cache
    assert caching_openai._async_cache._storage is cache

    # This assertion tests the llama-index-openai code directly; because we use a private method,
    # we want to notice if it changes in future versions, e.g. in order to change _filter_model_kwargs
    assert caching_openai._get_model_kwargs(foo=1) == {'model': 'gpt-4.1-mini', 'foo': 1, 'temperature': 0.0}
    assert caching_openai._async_http_client is openai_llm._async_http_client

    response = await caching_openai.acomplete('The capital of France is ')
    assert 'Paris' in response.text
    expected_cache_key = LLMCacheKey(messages=(), prompt='The capital of France is ', formatted=False, kwargs=frozendict({'model': 'gpt-4.1-mini', 'temperature': 0.0}))
    assert cache == { expected_cache_key: response }

    response2 = await caching_openai.acomplete('The capital of England is ')
    assert response2 is not response
    assert response2.text != response.text
    expected_cache_key2 = LLMCacheKey(messages=(), prompt='The capital of England is ', formatted=False, kwargs=frozendict({'model': 'gpt-4.1-mini', 'temperature': 0.0}))
    assert cache == { expected_cache_key: response, expected_cache_key2: response2 }

    # Mutate cached response, see that it's returned from cache
    response.text = '42'
    response3 = await caching_openai.acomplete('The capital of France is ')
    assert response3 is response
    assert response3.text == '42'

    # Test chat API
    response4 = await caching_openai.achat([ChatMessage('What is capital of France? Answer with its name only.')])
    expected_cache_key3 = LLMCacheKey(messages=(CanonicalizedChatMessage(ChatMessage(role=MessageRole.USER,
                                                                                     blocks=[TextBlock(text='What is capital of France? Answer with its name only.')])),),
                                      prompt=None, formatted=False, kwargs=frozendict({'model': 'gpt-4.1-mini', 'temperature': 0.0}))
    assert cache == { expected_cache_key: response, expected_cache_key2: response2, expected_cache_key3: response4 }

@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_responses_cache(openai_responses_llm: OpenAIResponses, disk_llm_cache: LLMCacheBackend) -> None:
    def assert_cache_contents(expected: dict[LLMCacheKey, CompletionResponse | ChatResponse]) -> None:
        assert len(disk_llm_cache) == len(expected)
        for k, v in expected.items():
            assert disk_llm_cache[k] == v

    caching_openai = CachingOpenAIResponses.adapt(openai_responses_llm, disk_llm_cache)

    assert caching_openai._cache is disk_llm_cache
    assert caching_openai._async_cache._storage is disk_llm_cache

    # This assertion tests the llama-index-openai code directly; because we use a private method,
    # we want to notice if it changes in future versions, e.g. in order to change _filter_model_kwargs
    assert caching_openai._get_model_kwargs(foo=1) == {'foo': 1,
                                                       'include': None,
                                                       'instructions': None,
                                                       'max_output_tokens': None,
                                                       'metadata': None,
                                                       'model': 'gpt-4.1-mini',
                                                       'previous_response_id': None,
                                                       'store': False,
                                                       'temperature': 0.0,
                                                       'tools': [],
                                                       'top_p': 1.0,
                                                       'truncation': 'disabled',
                                                       'user': None}
    assert caching_openai._async_http_client is openai_responses_llm._async_http_client

    default_kwargs = caching_openai._filter_model_kwargs(caching_openai._get_model_kwargs())

    response = await caching_openai.acomplete('The capital of France is ')
    assert 'Paris' in response.text
    expected_cache_key = LLMCacheKey(messages=(), prompt='The capital of France is ', formatted=False,
                                     kwargs=frozendict(default_kwargs))
    assert_cache_contents({ expected_cache_key: response })

    response2 = await caching_openai.acomplete('The capital of England is ')
    assert response2 is not response
    assert response2.text != response.text
    expected_cache_key2 = LLMCacheKey(messages=(), prompt='The capital of England is ', formatted=False,
                                      kwargs=frozendict(default_kwargs))
    assert_cache_contents({ expected_cache_key: response,
                             expected_cache_key2: response2 })

    response3 = await caching_openai.acomplete('The capital of France is ')
    assert response3 == response and response3 is not response, 'response returned from cache after serialization roundtrip'

    # Test chat API
    response4 = await caching_openai.achat([ChatMessage('What is capital of France? Answer with its name only.')])
    expected_cache_key3 = LLMCacheKey(messages=(CanonicalizedChatMessage(ChatMessage(role=MessageRole.USER,
                                                                                     blocks=[TextBlock(text='What is capital of France? Answer with its name only.')])),),
                                      prompt=None, formatted=False, kwargs=frozendict(default_kwargs))
    assert_cache_contents({ expected_cache_key: response,
                             expected_cache_key2: response2,
                             expected_cache_key3: response4 })

