from __future__ import annotations

import logging
import threading
import weakref
from abc import ABC, abstractmethod
from typing import override

import attrs
import httpx
from attrs import field, frozen
from llama_index.core.llms import LLM

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
    def from_spec(self, spec: LLMSpec, context: LLMContext) -> LLM | None: ...

    @abstractmethod
    def to_spec(self, model: LLM) -> LLMSpec | None: ...


class LLMSetupHook(ABC):
    """A signature for a user callback to further configure a newly created LLM instance."""
    
    @abstractmethod
    def __call__[T: LLM](self, model: T, context: LLMContext, spec: LLMSpec) -> T: ...

class DefaultLLMProvider(LLMProvider):
    """Provides the standard llama-index LLM types.
    
    Can raise ImportError if a spec requests a model from a provider whose package is not installed.
    """

    # TODO if we adopt this approach, add cases for the other llama-index LLM types.

    @override
    def from_spec(self, spec: LLMSpec, context: LLMContext) -> LLM | None:
        match spec.origin:
            case 'openai':
                try:
                    from llama_index.llms.openai import OpenAI, OpenAIResponses
                    # Prefer OpenAIResponses if a nonspecific type is requested
                    if spec.llm_type_matches(OpenAIResponses):
                        return OpenAIResponses(model=spec.model_name, http_client=context.httpx_client, async_http_client=context.httpx_async_client)
                    if spec.llm_type_matches(OpenAI):
                        return OpenAI(model=spec.model_name, http_client=context.httpx_client, async_http_client=context.httpx_async_client)
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
            if isinstance(model, OpenAI | OpenAIResponses):
                return LLMSpec(origin='openai', model_name=model.model, llm_type_str=type(model).__name__)
        except ImportError:
            pass

        return None

@frozen(eq=False, hash=False)
class LLMContext:
    """Instantiates LLM instances.
    
    This is configurable via registering providers that know how to create models
    and registering hooks that further configure model instances after they are created.
    """

    # These are needed to create (most) model instances.
    # The existence of an AsyncClient implies that we're in an asyncio context; this code cannot be used otherwise.
    httpx_client: httpx.Client
    httpx_async_client: httpx.AsyncClient

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
