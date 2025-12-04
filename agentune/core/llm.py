from __future__ import annotations

import logging
import threading
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast, override

import attrs
import httpx
from attrs import field, frozen
from frozendict import frozendict
from llama_index.core.base.llms.types import ChatResponse, CompletionResponse
from llama_index.core.llms import LLM

from agentune.core.llmcache.base import LLMCacheBackend, LLMCacheKey
from agentune.core.util.lrucache import LRUCache

_logger = logging.getLogger(__name__)


type Jsonable = str | int | float | bool | None |  list[Jsonable] | dict[str, Jsonable]


@frozen
class LLMSpec:
    """Serializable specification of a model (but not how or where to access it).

    If the arguments specified here (besides the origin and model name) conflict with overrides passed to the LLMProvider,
    an error is raised. There is no precedence currently defined between the two.

    Args:
        origin: the name of the API standard, e.g. 'openai' (even if hosted on Azure).
        model_name: the model to use, e.g. 'gpt-4o'.
                    Different APIs use different names for this parameter, including but not always 'model',
                    so we don't call this 'model' because that wouldn't automatically work with all providers.
        kwargs: keyword arguments specific to the LLM type. They can affect the llama-index LLM instance
                and, depending on the LLM type, underlying libraries such as the openai client instance.
                This class can't validate whether the arguments given really exist or have legal values,
                but trying to create an LLM instance with them will fail later.
                Arguments whose value is None are silently ignored.
    """
    origin: str
    model_name: str
    kwargs: frozendict[str, Jsonable] = frozendict()

    # We store all additional args in kwargs to keep the json format compatible.
    # When we want to declare a particular (optional) parameter for user convenience, we add a constructor parameter
    # and a @property accessor, like 'timeout' below.

    def __init__(self, origin: str, model_name: str,
                 timeout: float | None = None,
                 **kwargs: Jsonable) -> None:
        if kwargs.keys() == { 'kwargs' }:
            # When deserializing, because we don't declare a normal 'kwargs' argument in the ctor, the kwargs dict is wrapped
            kwargs = cast(dict[str, Jsonable], kwargs['kwargs'])
        if timeout is not None:
            kwargs['timeout'] = timeout
        self.__attrs_init__(origin, model_name, frozendict(kwargs)) # type: ignore[attr-defined] # mypy doesn't know about __attrs_init__

    @property
    def timeout(self) -> float | None:
        return cast(float | None, self.kwargs.get('timeout'))


@attrs.define
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

    def for_specs(self, filter: Callable[[LLMSpec], bool]) -> LLMProvider:
        """Return a provider that calls the original one only for specs that match the given predicate."""
        return FilteredLLMProvider(self, filter, '(filtered)')

    def for_models(self, *models: str) -> LLMProvider:
        """Return a provider that calls the original one only for model names that match the given list."""
        return FilteredLLMProvider(self, lambda spec: spec.model_name in models, f'(for models {', '.join(models)})')

    def _compose_kwargs(self, self_args: dict[str, Jsonable], spec: LLMSpec) -> dict[str, Any]:
        """Combine the parameters to this class with the parameters in the spec, ignoring any fields set to None.

        If both sides specify different values for the same parameter, an error is raised.
        """
        self_args = {k: v for k, v in self_args.items() if v is not None}
        spec_args = {k: v for k, v in spec.kwargs.items() if v is not None}
        for k in spec_args.keys() & self_args.keys():
            if spec_args[k] != self_args[k]:
                raise ValueError(f'Conflicting values for argument {k}: {spec_args[k]} in LLMSpec, {self_args[k]} in LLMProvider')
        return {**self_args, **spec_args}


@frozen(eq=False, hash=False)
class FilteredLLMProvider(LLMProvider):
    """A provider that calls the inner one only for specs that match the given predicate."""
    inner: LLMProvider
    filter: Callable[[LLMSpec], bool]
    filter_desc: str

    def from_spec(self, spec: LLMSpec, context: LLMContext) -> LLM | None:
        if self.filter(spec):
            return self.inner.from_spec(spec, context)
        else:
            return None

    def __str__(self) -> str:
        return f'{self.inner} {self.filter_desc}'


class FakeTransport(httpx.BaseTransport):
    """A transport that fails all requests."""
    @override 
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.RequestError('Synchronous HTTP requests disallowed')

# A synchronous httpx.Client that fails all requests. Used to prevent accidental use of synchronous HTTP requests.
fake_httpx_client: httpx.Client = httpx.Client(transport=FakeTransport())

@frozen(eq=False, hash=False)
class LLMContext:
    """Instantiates LLM instances.
    
    This is configurable via registering providers that know how to create models
    and registering hooks that further configure model instances after they are created.

    Args:
        fail_if_cannot_cache: if a cache_backend is provided, but caching is not supported or implement for the model
                              being requested, fail if this is True, and proceed without caching if this is False.
    """

    # This is needed to create (most) model instances.
    # The existence of an AsyncClient implies that we're in an asyncio context; this code cannot be used otherwise.
    httpx_async_client: httpx.AsyncClient
    providers: tuple[LLMProvider, ...]

    httpx_client: httpx.Client = fake_httpx_client # Disallow synchronous HTTP requests by default

    cache_backend: LLMCacheBackend | None = field(factory=lambda: LRUCache[LLMCacheKey, CompletionResponse | ChatResponse](1000))
    fail_if_cannot_cache: bool = True

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
                return llm
        raise ValueError(f'No provider found for spec {spec}')
