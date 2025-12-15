import gc
import json
import logging
import os
from typing import cast

import attrs
import httpx
import openai
import pytest
from llama_index.llms.openai import OpenAI, OpenAIResponses

from agentune.api.base import RunContext
from agentune.api.defaults import create_default_httpx_async_client
from agentune.core.llm import LLMContext, LLMSpec
from agentune.core.openai import CachingOpenAI, CachingOpenAIResponses, OpenAIProvider
from agentune.core.sercontext import SerializationContext
from agentune.core.util.lrucache import LRUCache

_logger = logging.getLogger(__name__)

async def test_openai_provider(httpx_async_client: httpx.AsyncClient) -> None:
    context1 = LLMContext(httpx_async_client, providers = (OpenAIProvider(),))
    context2 = LLMContext(httpx_async_client, providers = (OpenAIProvider(use_responses_api=False),))

    llm_spec = LLMSpec('openai', 'gpt-4o')
    llm = context1.from_spec(llm_spec)
    assert isinstance(llm, OpenAIResponses)
    assert llm.model == 'gpt-4o'

    llm2 = context2.from_spec(llm_spec)
    assert isinstance(llm2, OpenAI)
    assert llm2.model == 'gpt-4o'

    llm3 = context1.from_spec(LLMSpec('openai', 'nonesuch'))
    assert isinstance(llm3, OpenAIResponses)
    assert llm3.model == 'nonesuch', 'Willing to instantiate unfamiliar models'

    with pytest.raises(ValueError, match='No provider found for spec'):
        context1.from_spec(LLMSpec('closedai', 'gpt-4o'))

async def test_llmprovider_filter(httpx_async_client: httpx.AsyncClient) -> None:
    context1 = LLMContext(httpx_async_client, providers = (OpenAIProvider().for_models('gpt-4o', 'gpt-4.1'),))
    assert cast(OpenAIResponses, context1.from_spec(LLMSpec('openai', 'gpt-4o'))).model == 'gpt-4o'
    assert cast(OpenAIResponses, context1.from_spec(LLMSpec('openai', 'gpt-4.1'))).model == 'gpt-4.1'
    with pytest.raises(ValueError, match='No provider found for spec'):
        context1.from_spec(LLMSpec('openai', 'gpt-5'))

    context2 = LLMContext(httpx_async_client, providers = (
        OpenAIProvider(use_responses_api=True).for_models('gpt-4o', 'gpt-4.1'),
        OpenAIProvider(use_responses_api=False)
    ))
    llm1 = context2.from_spec(LLMSpec('openai', 'gpt-4o'))
    assert isinstance(llm1, OpenAIResponses)
    assert llm1.model == 'gpt-4o'
    llm2 = context2.from_spec(LLMSpec('openai', 'gpt-4.1'))
    assert isinstance(llm2, OpenAIResponses)
    assert llm2.model == 'gpt-4.1'
    llm3 = context2.from_spec(LLMSpec('openai', 'gpt-5'))
    assert isinstance(llm3, OpenAI)
    assert llm3.model == 'gpt-5'


async def test_llm_instance_cache(llm_context: LLMContext) -> None:
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
    llm_context = LLMContext(httpx_async_client, (OpenAIProvider(),), cache_backend=LRUCache(1000))

    llm_spec = LLMSpec('openai', 'gpt-4o')

    llm = llm_context.from_spec(llm_spec)
    assert isinstance(llm, CachingOpenAIResponses)
    assert llm.model == 'gpt-4o'

    llm_context2 = LLMContext(httpx_async_client, (OpenAIProvider(use_responses_api=False),), cache_backend=LRUCache(1000))

    llm_spec3 = LLMSpec('openai', 'gpt-4o')
    llm3 = llm_context2.from_spec(llm_spec3)
    assert isinstance(llm3, CachingOpenAI)
    assert llm3.model == 'gpt-4o'

    llm4 = llm_context.from_spec(LLMSpec('openai', 'nonesuch'))
    assert isinstance(llm4, CachingOpenAIResponses)
    assert llm4.model == 'nonesuch', 'Willing to instantiate unfamiliar models'

    with pytest.raises(ValueError, match='No provider found for spec'):
        llm_context.from_spec(LLMSpec('closedai', 'gpt-4o'))

async def test_llmspec_serialization(ser_context: SerializationContext) -> None:
    spec = LLMSpec('openai', 'gpt-4o')
    assert ser_context.converter.loads(ser_context.converter.dumps(spec), LLMSpec) == spec

    spec = LLMSpec('openai', 'gpt-4o', timeout=60.0) # type:ignore[call-arg] # mypy doesn't understand attrs with custom init
    assert json.loads(ser_context.converter.dumps(spec)) ==  {'origin': 'openai', 'model_name': 'gpt-4o', 'kwargs': { 'timeout': 60.0} }
    assert ser_context.converter.loads(ser_context.converter.dumps(spec), LLMSpec) == spec

    spec = LLMSpec('openai', 'gpt-4o', foo='bar') # type:ignore[call-arg] # mypy doesn't understand attrs with custom init
    assert json.loads(ser_context.converter.dumps(spec)) ==  {'origin': 'openai', 'model_name': 'gpt-4o', 'kwargs': { 'foo': 'bar' }}
    assert ser_context.converter.loads(ser_context.converter.dumps(spec), LLMSpec) == spec

async def test_llm_env_vars() -> None:
    orig_key = os.environ.get('OPENAI_API_KEY')
    orig_url = os.environ.get('OPENAI_API_BASE')
    if orig_url is not None:
        del os.environ['OPENAI_API_BASE']
    if orig_key is not None:
        del os.environ['OPENAI_API_KEY']

    try:

        async with await RunContext.create(llm_providers=OpenAIProvider()) as ctx:
            llm = ctx.llm.get(LLMSpec('openai', 'gpt-4o'))
            assert isinstance(llm, OpenAIResponses) # To make mypy happy
            assert llm.api_base == 'https://api.openai.com/v1', 'Default api_base is used'
            assert llm.api_key == '', 'API key remains unset'

        os.environ['OPENAI_API_KEY'] = 'mykey'
        async with await RunContext.create(llm_providers=OpenAIProvider()) as ctx:
            llm = ctx.llm.get(LLMSpec('openai', 'gpt-4o'))
            assert isinstance(llm, OpenAIResponses) # To make mypy happy
            assert llm.api_base == 'https://api.openai.com/v1', 'Default api_base is used'
            assert llm.api_key == 'mykey', 'API key env var was used'

        os.environ['OPENAI_API_BASE'] = 'https://api.mycompany.com/v1'
        async with await RunContext.create(llm_providers=OpenAIProvider()) as ctx:
            llm = ctx.llm.get(LLMSpec('openai', 'gpt-4o'))
            assert isinstance(llm, OpenAIResponses) # To make mypy happy
            assert llm.api_base == 'https://api.mycompany.com/v1', 'Base url env var was used'
            assert llm.api_key == 'mykey', 'API key env var was used'

        async with await RunContext.create(llm_providers=OpenAIProvider(base_url='https://new.url/', api_key='mykey2')) as ctx:
            llm = ctx.llm.get(LLMSpec('openai', 'gpt-4o'))
            assert isinstance(llm, OpenAIResponses) # To make mypy happy
            assert llm.api_base == 'https://new.url/', 'Base url arg was used'
            assert llm.api_key == 'mykey2', 'API key arg was used'

    finally:
        if orig_url is not None:
            os.environ['OPENAI_API_BASE'] = orig_url
        else:
            del os.environ['OPENAI_API_BASE']
        if orig_key is not None:
            os.environ['OPENAI_API_KEY'] = orig_key
        else:
            del os.environ['OPENAI_API_KEY']

async def test_llm_url_api_key_params_are_used() -> None:
    provider = OpenAIProvider(base_url='http://nonesuch.domain/', api_key='nonesuch', max_retries=0) # type:ignore[call-arg] # mypy doesn't understand attrs with custom init
    async with await RunContext.create(llm_providers=provider) as ctx:
        llm = ctx.llm.get(LLMSpec('openai', 'gpt-4o'))
        assert isinstance(llm, OpenAIResponses) # To make mypy happy
        assert llm.api_base == 'http://nonesuch.domain/'
        assert llm.api_key == 'nonesuch'
        assert llm.max_retries == 0

        with pytest.raises(openai.APIConnectionError):
            await llm.acomplete('The capital of France is')

    provider = OpenAIProvider(api_key='nonesuch', max_retries=0) # type:ignore[call-arg] # mypy doesn't understand attrs with custom init
    async with await RunContext.create(llm_providers=provider) as ctx:
        llm = ctx.llm.get(LLMSpec('openai', 'gpt-4o'))
        with pytest.raises(openai.AuthenticationError):
            await llm.acomplete('The capital of France is')

async def test_llm_provider_and_spec_combine_params() -> None:
    provider = OpenAIProvider(timeout=10.0)
    async with await RunContext.create(llm_providers=provider,
                                       httpx_async_client=create_default_httpx_async_client(timeout=httpx.Timeout(30.0))) as ctx:
        spec = LLMSpec('openai', 'gpt-4o', temperature=0.5) # type:ignore[call-arg] # mypy doesn't understand attrs with custom init
        llm = ctx.llm.get(spec)
        assert isinstance(llm, OpenAIResponses) # To make mypy happy
        assert llm.timeout == 10.0, 'Arg from provider'
        assert llm.temperature == 0.5, 'Arg from spec'

        spec2 = LLMSpec('openai', 'gpt-4o', timeout=20.0) # type:ignore[call-arg] # mypy doesn't understand attrs with custom init
        with pytest.raises(ValueError, match='Conflicting values for argument timeout'):
            ctx.llm.get(spec2)

        spec3 = LLMSpec('openai', 'gpt-4o', nonesuch='value') # type:ignore[call-arg] # mypy doesn't understand attrs with custom init
        ctx.llm.get(spec3) # Arguments that don't match a real LLM parameter are silently ignored
        assert not hasattr(llm, 'nonesuch')
