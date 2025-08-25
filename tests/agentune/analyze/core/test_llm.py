import gc
import logging

import attrs
import httpx
import pytest
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI, OpenAIResponses

from agentune.analyze.core.llm import LLMContext, LLMSpec
from agentune.analyze.llmcache.openai_cache import CachingOpenAI, CachingOpenAIResponses
from agentune.analyze.util.lrucache import LRUCache

_logger = logging.getLogger(__name__)

async def test_llm_context(httpx_async_client: httpx.AsyncClient) -> None:
    llm_context = LLMContext(httpx_async_client, cache_backend=None)

    llm_spec = LLMSpec('openai', 'gpt-4o')
    assert llm_spec.llm_type_str == 'LLM'
    assert llm_spec.llm_type_matches(LLM)
    assert llm_spec.llm_type_matches(OpenAI)
    assert llm_spec.llm_type_matches(OpenAIResponses)

    llm = llm_context.from_spec(llm_spec)
    assert isinstance(llm, OpenAIResponses)
    assert llm.model == 'gpt-4o'

    llm_spec2 = llm_context.to_spec(llm)
    assert llm_spec2 == LLMSpec('openai', 'gpt-4o', 'OpenAIResponses')

    llm_spec3 = LLMSpec('openai', 'gpt-4o', 'OpenAI')
    llm3 = llm_context.from_spec(llm_spec3)
    assert isinstance(llm3, OpenAI)
    assert llm3.model == 'gpt-4o'
    assert llm_context.to_spec(llm3) == llm_spec3

    llm4 = llm_context.from_spec(LLMSpec('openai', 'nonesuch'))
    assert isinstance(llm4, OpenAIResponses)
    assert llm4.model == 'nonesuch', 'Willing to instantiate unfamiliar models'

    with pytest.raises(ValueError, match='No provider found for spec'):
        llm_context.from_spec(LLMSpec('closedai', 'gpt-4o'))

async def test_llm_instance_cache(httpx_async_client: httpx.AsyncClient) -> None:
    llm_context = LLMContext(httpx_async_client, cache_backend=None)

    llm_spec = LLMSpec('openai', 'gpt-4o')
    llm = llm_context.from_spec(llm_spec)
    llm_spec2 = attrs.evolve(llm_spec)
    llm2 = llm_context.from_spec(llm_spec)
    assert llm_spec is not llm_spec2
    assert llm is llm2, 'Cached instance is returned'

    llm_instance_id = id(llm)

    del llm
    del llm2
    gc.collect()

    llm3 = llm_context.from_spec(llm_spec)
    assert id(llm3) != llm_instance_id, 'New instance is created'

async def test_llm_context_with_cache(httpx_async_client: httpx.AsyncClient) -> None:
    llm_context = LLMContext(httpx_async_client, cache_backend=LRUCache(1000))

    llm_spec = LLMSpec('openai', 'gpt-4o')
    assert llm_spec.llm_type_matches(CachingOpenAI)
    assert llm_spec.llm_type_matches(CachingOpenAIResponses)

    llm = llm_context.from_spec(llm_spec)
    assert isinstance(llm, CachingOpenAIResponses)
    assert llm.model == 'gpt-4o'

    llm_spec2 = llm_context.to_spec(llm)
    assert llm_spec2 == LLMSpec('openai', 'gpt-4o', 'OpenAIResponses')

    llm_spec3 = LLMSpec('openai', 'gpt-4o', 'OpenAI')
    llm3 = llm_context.from_spec(llm_spec3)
    assert isinstance(llm3, CachingOpenAI)
    assert llm3.model == 'gpt-4o'
    assert llm_context.to_spec(llm3) == llm_spec3

    llm4 = llm_context.from_spec(LLMSpec('openai', 'nonesuch'))
    assert isinstance(llm4, CachingOpenAIResponses)
    assert llm4.model == 'nonesuch', 'Willing to instantiate unfamiliar models'

    with pytest.raises(ValueError, match='No provider found for spec'):
        llm_context.from_spec(LLMSpec('closedai', 'gpt-4o'))
