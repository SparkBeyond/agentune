import logging

import httpx
from attrs import frozen
from llama_index.llms.openai import OpenAIResponses

from agentune.analyze.core.llm import LLMContext, LLMSpec
from agentune.analyze.core.sercontext import LLMWithSpec, SerializationContext

_logger = logging.getLogger(__name__)

async def test_llm_serialization(httpx_async_client: httpx.AsyncClient) -> None:
    llm_context = LLMContext(httpx_async_client)
    serialization_context = SerializationContext(llm_context)

    llm_spec = LLMSpec('openai', 'gpt-4o')
    llm = llm_context.from_spec(llm_spec)

    @frozen
    class LlmUsingFoobar:
        model: LLMWithSpec

    foobar = LlmUsingFoobar(LLMWithSpec(llm_spec, llm))

    serialized = serialization_context.converter.unstructure(foobar)
    recovered = serialization_context.converter.structure(serialized, LlmUsingFoobar)
    assert recovered.model.spec == llm_spec
    assert isinstance(recovered.model.llm, OpenAIResponses)
    assert recovered.model.llm.model == 'gpt-4o'

