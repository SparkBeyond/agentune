from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, override

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import LLM

from agentune.analyze.llmcache.base import CachingLLMMixin, LLMCacheKey

# Use conditional imports because we don't want to add a hard dependency on every llama-index llm provider
try:
    from llama_index.llms.openai import OpenAI, OpenAIResponses

    if TYPE_CHECKING:
        from typing import Protocol

        class OpenaiLikeLLM(Protocol):
            # Defined in both OpenAI and OpenAIResponses
            # The returned value includes the relevant attributes of `self`, such as temperature.
            def _get_model_kwargs(self, **kwargs: Any) -> dict[str, Any]: ...
    else:
        OpenaiLikeLLM = LLM

    class CachingOpenAIBase(CachingLLMMixin, OpenaiLikeLLM):
        def _filter_model_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
            """Any arguments that should NOT be included in the cache key should be blacklisted here.
            By default (without extra blacklisting effort) this may include some arguments that don't have to be
            part of the cache key, but don't really hurt caching either.
            """
            # Some of these only exist in the chat API or only in the responses API; I didn't bother to separate them
            for key in ['background', 'prompt_cache_key', 'safety_identifier', 'service_tier', 'store', 'user']:
                kwargs.pop(key, None)
            return kwargs

        @override
        def _chat_key(self, messages: Sequence[ChatMessage], **kwargs: Any) -> LLMCacheKey:
            model_kwargs = self._filter_model_kwargs(self._get_model_kwargs(**kwargs))
            return LLMCacheKey(tuple(messages), None, False, model_kwargs)

        @override
        def _completion_key(self, prompt: str, formatted: bool = False, **kwargs: Any) -> LLMCacheKey:
            model_kwargs = self._filter_model_kwargs(self._get_model_kwargs(**kwargs))
            return LLMCacheKey((), prompt, formatted, model_kwargs)

    class CachingOpenAI(CachingOpenAIBase, OpenAI):
        pass

    class CachingOpenAIResponses(CachingOpenAIBase, OpenAIResponses):
        pass


except ImportError:
    pass

