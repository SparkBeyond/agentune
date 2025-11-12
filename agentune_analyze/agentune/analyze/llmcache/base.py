"""An API for caching (some) calls to LLMs of (some) providers.

Caching is enabled separately for each provider (= subclass of LLM), using provider-specific cache key classes
that declare the provider-specific parameters of methods like LLM.chat that should be part of the cache key.

This package implements only the bare minimum needed for our own uses. That means it only declares cache key parameters
that we actually pass to models. (It would be a large and never-ending task to document and test all parameters
accepted by e.g. OpenAI models that ought to affect caching.)

If another parameter is passed (or set as a default on an LLM instance), e.g. by the user supplying custom LLM configurations
to LLMContext, which affects LLM responses and so should participate in LLM cache keys, it will effectively break caching,
or at least make cached responses unshareable with runs that don't use that custom configuration.
"""
from __future__ import annotations

import hashlib
import logging
from abc import abstractmethod
from collections.abc import Sequence
from functools import cached_property
from typing import Any, Self, cast, override

import pydantic
from attrs import field, frozen
from frozendict import frozendict
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
)
from llama_index.core.llms import LLM

from agentune.analyze.llmcache.impl import (
    CanonicalizedChatMessage,
    _kwargs_freeze_converter,
    _messages_tuple_converter,
)
from agentune.analyze.util.asyncmap import HalfAsyncMap, KVStore

_logger = logging.getLogger(__name__)


@frozen
class LLMCacheKey:
    """Cache key for LLM calls.

    Args:
        messages: used for chat API
        prompt, formatted: used for completion API
        kwargs: used for provider-specific parameters
    """
    messages: tuple[CanonicalizedChatMessage, ...] = field(converter=_messages_tuple_converter)
    prompt: str | None
    formatted: bool
    kwargs: frozendict[str, Any] = field(converter=_kwargs_freeze_converter)

    long_hash: bytes = field(init=False, eq=False, repr=False)
    @long_hash.default
    def _long_hash_init(self) -> bytes:
        """A 32-byte hash which is longer and has fewer collisions than __hash__()."""
        return hashlib.sha256(str(self).encode()).digest()

    _hash: int = field(init=False, eq=False, repr=False)
    @_hash.default
    def _hash_init(self) -> int:
        # Try to hash self, fallback to the expensive long_hash if it fails.
        # kwargs can contain unhashable values like lists and dicts, and _kwargs_freeze_converter() tries to take care of them
        # but it's better to have a fallback than to rely on that.
        try:
            return hash(frozendict({'messages': self.messages, 'prompt': self.prompt, 'formatted': self.formatted, 'kwargs': self.kwargs}))
        except TypeError:
            _logger.warning(f'Unhashable cache key {self}, falling back to long_hash')
            return hash(self.long_hash)

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        """Stringification that preserves equality, suitable for use as a cache key"""
        messages = tuple(m.message.model_dump(warnings='error') for m in self.messages)
        sorted_kwargs = { k: self.kwargs[k] for k in sorted(self.kwargs) }
        return f'{messages=}, {self.prompt=}, {self.formatted=}, {sorted_kwargs=}'


# NOTE that the LLM mixins and subclasses are not attrs dataclasses, they are Pydantic models, like the base class LLM.

type LLMCacheBackend = KVStore[LLMCacheKey, CompletionResponse | ChatResponse]

class CachingLLMMixin(LLM):
    """Mixin for an LLM subclass adding caching behavior for the methods declared in LLM.
    This class is abstract; only LLM-provider-specific subclasses can be concrete.

    Concurrent calls to an async method like `achat` with the same parameters join into a single call
    to the wrapped LLM, even though the result is not cached yet.

    Concurrent calls to *synchronous* methods like `chat`, and concurrent calls to equivalent methods like `chat` and `achat`
    with the same parameters, are *not* deduplicated. (We don't use LLMs synchronously so we don't need this.)
    However, the results of synchronous calls *are* are cached and can then provide responses for both sync and async calls.

    The chat_cache and completion_cache can be the same if a single cache supports storing both value types;
    we use separate fields to avoid casts.

    Streaming methods are not supported and calling them will raise an error to make it clear they're not cached.
    (We can add support if we ever need it.)
    """

    _cache: LLMCacheBackend = pydantic.PrivateAttr()

    def __init__(self, cache: LLMCacheBackend, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cache = cache

    @cached_property
    def _async_cache(self) -> HalfAsyncMap[LLMCacheKey, ChatResponse | CompletionResponse]:
        return HalfAsyncMap(storage=self._cache)

    @abstractmethod
    def _chat_key(self, messages: Sequence[ChatMessage], **kwargs: Any) -> LLMCacheKey: ...

    @abstractmethod
    def _completion_key(self, prompt: str, formatted: bool = False, **kwargs: Any) -> LLMCacheKey: ...

    @override
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        key = self._chat_key(messages, **kwargs)
        match self._cache.get(key):
            case ChatResponse() as response: return response
            case None:
                result = super().chat(messages, **kwargs) # type: ignore[safe-super]
                self._cache[key] = result
                return result
            case other:
                raise ValueError(f'Unexpected cache value type {type(other)}')

    @override
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        key = self._chat_key(messages, **kwargs)
        result = await self._async_cache.get_or_update(key, lambda: super(CachingLLMMixin, self).achat(messages, **kwargs)) # type: ignore[safe-super]
        return cast(ChatResponse, result)

    @override
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        key = self._completion_key(prompt, formatted, **kwargs)
        match self._cache.get(key):
            case CompletionResponse() as response: return response
            case None:
                result = super().complete(prompt, formatted, **kwargs) # type: ignore[safe-super]
                self._cache[key] = result
                return result
            case other:
                raise ValueError(f'Unexpected cache value type {type(other)}')

    @override
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        key = self._completion_key(prompt, formatted, **kwargs)
        result = await self._async_cache.get_or_update(key, lambda: super(CachingLLMMixin, self).acomplete(prompt, formatted, **kwargs)) # type: ignore[safe-super]
        return cast(CompletionResponse, result)

    @override
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        raise NotImplementedError('Caching streaming LLM APIs is not implemented')

    @override
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        raise NotImplementedError('Caching streaming LLM APIs is not implemented')

    @override
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError('Caching streaming LLM APIs is not implemented')

    @override
    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        raise NotImplementedError('Caching streaming LLM APIs is not implemented')

    @classmethod
    def adapt(cls, llm: LLM, cache: LLMCacheBackend) -> Self:
        """Create a caching equivalent of the given LLM."""
        params = dict(llm)
        result = cls(**params, cache=cache)
        # Copy params that need to be copied even though they're not marked as serializable, e.g. http_client
        # This needs to be validated per LLM implementation that we support.
        for name in llm.__private_attributes__:
            setattr(result, name, getattr(llm, name))
        return result
