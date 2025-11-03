from __future__ import annotations

import logging
import threading
import weakref
from abc import ABC, abstractmethod
from typing import override

import attrs
import httpx
from attrs import field, frozen
from llama_index.core.base.llms.types import ChatResponse, CompletionResponse
from llama_index.core.llms import LLM

from agentune.analyze.llmcache.base import LLMCacheBackend, LLMCacheKey
from agentune.analyze.llmcache.openai_cache import CachingOpenAI, CachingOpenAIResponses
from agentune.analyze.util.lrucache import LRUCache

_logger = logging.getLogger(__name__)


@frozen
class LLMSpec:
    """Serializable specification of a model (but not how or where to access it)."""
    origin: str 
    # Logical provider of the model, e.g. 'openai' (even if hosted on azure).
    # This lets us instantiate models even if we're not familiar with the model name.
    model_name: str
    llm_type_str: str = LLM.__name__
    # Local name of a subclass of class LLM (e.g. output of `type(llm).__name__`).
    # Can be used to demand a subtype of class LLM, whether generic (eg FunctionCallingLLM) or specific (eg OpenAI).

    def llm_type_matches[T](self, model: type) -> bool:
        return any(tpe.__name__ == self.llm_type_str for tpe in model.mro())
 

class LLMProvider(ABC):
    """Converts between LLM instances and LLMSpecs in both directions.
    
    If an exception is raised by one of the methods, the process fails and no other providers are tried.
    """

    @abstractmethod
    def from_spec(self, spec: LLMSpec, context: LLMContext) -> LLM | None:
        """Provide an LLM instance matching a spec.

        If a cache_backend is provided, try to use it; if fail_if_cannot_cache is True,
        raise a TypeError if the model can be supported but not with caching.
        """
        ...

    @abstractmethod
    def to_spec(self, model: LLM) -> LLMSpec | None: ...


class LLMSetupHook(ABC):
    """A signature for a user callback to further configure a newly created LLM instance."""
    
    @abstractmethod
    def __call__[T: LLM](self, model: T, context: LLMContext, spec: LLMSpec) -> T: ...

class FakeTransport(httpx.BaseTransport):
    """A transport that fails all requests."""
    @override 
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.RequestError('Synchronous HTTP requests disallowed')

# A synchronous httpx.Client that fails all requests. Used to prevent accidental use of synchronous HTTP requests.
fake_httpx_client: httpx.Client = httpx.Client(transport=FakeTransport())


class DefaultLLMProvider(LLMProvider):
    """Provides the standard llama-index LLM types.
    
    Can raise ImportError if a spec requests a model from a provider whose package is not installed.
    """

    @override
    def from_spec(self, spec: LLMSpec, context: LLMContext) -> LLM | None:
        match context.httpx_async_client.timeout.read:
            case None: read_timeout = 60.0*60*24 # None means no timeout in httpx; set a very high limit but not literally infinite
            case timeout: read_timeout = float(timeout)

        match spec.origin:
            case 'openai':
                try:
                    from llama_index.llms.openai import OpenAI, OpenAIResponses
                    # Prefer OpenAIResponses if a nonspecific type is requested
                    llm: LLM
                    if spec.llm_type_matches(OpenAIResponses):
                        llm = OpenAIResponses(model=spec.model_name, http_client=context.httpx_client,
                                              async_http_client=context.httpx_async_client,
                                              timeout=read_timeout)
                        if context.cache_backend is not None:
                            return CachingOpenAIResponses.adapt(llm, context.cache_backend)
                        return llm
                    if spec.llm_type_matches(OpenAI):
                        llm = OpenAI(model=spec.model_name, http_client=context.httpx_client,
                                     async_http_client=context.httpx_async_client,
                                     timeout=read_timeout)
                        if context.cache_backend is not None:
                            return CachingOpenAI.adapt(llm, context.cache_backend)
                        return llm
                    raise ValueError(f'Spec for "openai" model has unsatisfiable llm type {spec.llm_type_str}')
                except ImportError as e:
                    e.add_note('Install the llama-index-llms-openai package to use this model.')
                    raise
            case _:
                return None
    
    @override
    def to_spec(self, model: LLM) -> LLMSpec | None:
        try:
            from llama_index.llms.openai import OpenAI, OpenAIResponses
            if isinstance(model, OpenAI):
                return LLMSpec(origin='openai', model_name=model.model, llm_type_str=OpenAI.__name__)
            elif isinstance(model, OpenAIResponses):
                return LLMSpec(origin='openai', model_name=model.model, llm_type_str=OpenAIResponses.__name__)
        except ImportError:
            pass

        return None

@frozen(eq=False, hash=False)
class LLMContext:
    """Instantiates LLM instances.
    
    This is configurable via registering providers that know how to create models
    and registering hooks that further configure model instances after they are created.

    Args:
        fail_if_cannot_cache: if a cache_backend is provided, but caching is not supported or implement for the model
                              being requested, fail if this is True, and proceed without caching if this is False.
    """

    # These are needed to create (most) model instances.
    # The existence of an AsyncClient implies that we're in an asyncio context; this code cannot be used otherwise.
    httpx_async_client: httpx.AsyncClient

    httpx_client: httpx.Client = fake_httpx_client # Disallow synchronous HTTP requests by default

    cache_backend: LLMCacheBackend | None = field(factory=lambda: LRUCache[LLMCacheKey, CompletionResponse | ChatResponse](1000))
    fail_if_cannot_cache: bool = True

    hooks: tuple[LLMSetupHook, ...] = ()
    providers: tuple[LLMProvider, ...] = field(factory=lambda: (DefaultLLMProvider(),))

    # Weakly cache LLM instances so we don't e.g. create an instance per Feature unnecessarily.
    # init=False ensures copies of the context (e.g. with different providers or hooks) don't share the cache.
    # Only access the cache while holding the lock.
    # Because it's a weak-value cache, we don't bother enforcing a max size.
    # WeakValueDictionary only removes entries the next time it is accessed; the failure mode here would be 
    # to create very many LLM instances and then GC them all, keeping the cache full of entries pointing nowhere,
    # but this is an edge case and it would still be small compared to the size of the LLM instances themselves.
    cache: weakref.WeakValueDictionary[LLMSpec, LLM] = field(init=False, factory=weakref.WeakValueDictionary)
    cache_lock: threading.Lock = field(init=False, factory=threading.Lock)

    def with_provider(self, provider: LLMProvider) -> LLMContext:
        return attrs.evolve(self, providers=(*self.providers, provider))

    def without_provider(self, provider: LLMProvider) -> LLMContext:
        return attrs.evolve(self, providers=tuple(p for p in self.providers if p != provider))

    def with_hook(self, hook: LLMSetupHook) -> LLMContext:
        return attrs.evolve(self, hooks=(*self.hooks, hook))

    def from_spec(self, spec: LLMSpec) -> LLM:
        with self.cache_lock:
            ret = self.cache.get(spec)
            if ret:
                return ret
        created = self._from_spec_uncached(spec)
        with self.cache_lock:
            self.cache[spec] = created
        return created

    def _from_spec_uncached(self, spec: LLMSpec) -> LLM:    
        for provider in self.providers:
            llm = provider.from_spec(spec, self)
            if llm is not None:
                for hook in self.hooks:
                    llm = hook(llm, self, spec)
                return llm
        raise ValueError(f'No provider found for spec {spec}')

    def to_spec(self, model: LLM) -> LLMSpec: 
        for provider in self.providers:
            spec = provider.to_spec(model)
            if spec is not None:
                return spec
        raise ValueError(f'No provider found for model type {type(model)}')
