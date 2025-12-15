from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, override

import httpx
from attrs import field, frozen
from frozendict import frozendict
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI, OpenAIResponses

from agentune.core.llm import LLMContext, LLMProvider, LLMSpec
from agentune.core.llmcache.base import CachingLLMMixin, LLMCacheKey
from agentune.core.util.attrutil import frozendict_converter

# We currently have a dependency on llama-index-llms-openai, and so the imports are global.
# In the future, when we support other providers like Anthropic, we can make those class definitions dependent
# on whether the imports succeed and keep the dependency optional, and do the same for openai.

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

@frozen(eq=False, hash=False)
class OpenAIProvider(LLMProvider):
    """Provisions OpenAI LLM instances.

    If the arguments specified here (besides use_responses_api) conflict with same-name arguments in the LLMSpec,
    an error is raised. There is no precedence currently defined between the two.

    Args:
        use_responses_api: if True, use the newer Responses API; if False, use older completions-based APIs.

        base_url: base URL for the OpenAI API. If None, uses the environment variable OPENAI_API_BASE.
        api_key: secret key, or a callable returning that key (in case of a refreshing token).
                 If None, uses the environment variable OPENAI_API_KEY.
        timeout: the read timeout for API requests. If None, uses the default timeout defined at the RunContext level.

        api_version: which API version to assume when building the request.
                     This is not required for newer service provideres that support the OpenAI v1 API.

        kwargs: keyword arguments that will be passed as-is to the library client.
                They can affect the llama-index OpenAI LLM instance or the underling openai client.
                Any fields explicitly declared in this class will override values given in kwargs.
    """
    # Our own params
    use_responses_api: bool = True

    # openai client params
    base_url: str | httpx.URL | None = None
    api_key: str | Callable[[], Awaitable[str]] | None = None
    timeout: float | None = None

    # llama-index openai params
    api_version: str | None = None

    # Open-ended
    kwargs: frozendict[str, Any] = field(default=frozendict(), converter=frozendict_converter)

    # Manually defined init so that the kwargs can be passed as **kwargs
    def __init__(self,
                 use_responses_api: bool = True,
                 base_url: str | httpx.URL | None = None,
                 api_key: str | Callable[[], Awaitable[str]] | None = None,
                 timeout: float | None = None,
                 api_version: str | None = None,
                 **kwargs: Any) -> None:
        self.__attrs_init__(use_responses_api, base_url, api_key, timeout, api_version, frozendict(kwargs)) # type: ignore[attr-defined] # mypy doesn't know about __attrs_init__'


    @override
    def from_spec(self, spec: LLMSpec, context: LLMContext) -> LLM | None:
        kwargs = dict(self.kwargs)
        kwargs.update({
            'api_base': self.base_url, # NOTE different name. api_base is the llama-index name but we use the more common openai name.
            'api_key': self.api_key,
            'timeout': self.timeout,
            'api_version': self.api_version
        })
        kwargs = self._compose_kwargs(kwargs, spec)

        match spec.origin:
            case 'openai':
                if self.use_responses_api:
                    llm: LLM = OpenAIResponses(model=spec.model_name, http_client=context.httpx_client,
                                               async_http_client=context.httpx_async_client,
                                               **kwargs)
                    if context.cache_backend is not None:
                        return CachingOpenAIResponses.adapt(llm, context.cache_backend)
                    return llm
                else:
                    llm = OpenAI(model=spec.model_name, http_client=context.httpx_client,
                                 async_http_client=context.httpx_async_client,
                                 **kwargs)
                    if context.cache_backend is not None:
                        return CachingOpenAI.adapt(llm, context.cache_backend)
                    return llm
            case _:
                return None
