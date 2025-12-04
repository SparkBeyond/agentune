from attrs import frozen
from llama_index.core.llms import LLM

from agentune.api.base import RunContext
from agentune.core.llm import LLMSpec
from agentune.core.llmcache.base import LLMCacheBackend
from agentune.core.sercontext import LLMWithSpec


@frozen
class BoundLlm:
    """Methods for accessing and configuring LLM access, bound to a RunContext instance."""
    run_context: RunContext

    def get_with_spec(self, spec: LLMSpec) -> LLMWithSpec:
        """Convert an LLMSpec to an LLM instance.

        An LLMSpec defines the (logical) provider and model to use, e.g. openai/gpt-4o.
        An LLM instance is a concrete implementation (from the llama-index library) exposing that model,
        using the LLM provider, authentication and caching settings of the context.

        LLMWithSpec is a class combining LLM and LLMSpec. It has the special property that,
        when serialized to JSON, only the LLMSpec is written; and when deserialized, an LLMWithSpec is restored
        with the same or equivalent LLM instance.
        """
        return LLMWithSpec(spec, self.get(spec))

    def get(self, spec: LLMSpec) -> LLM:
        """Convert an LLMSpec to an LLM instance.

        An LLMSpec defines the (logical) provider and model to use, e.g. openai/gpt-4o.
        An LLM instance is a concrete implementation (from the llama-index library) exposing that model,
        using the LLM provider, authentication and caching settings of the context.
        """
        return self.run_context._llm_context.from_spec(spec)

    @property
    def cache_backend(self) -> LLMCacheBackend | None:
        """Return the configured LLM cache backend (i.e. storage), if any."""
        return self.run_context._llm_context.cache_backend

    def clear_cache(self) -> None:
        """If an LLM cache is configured, clear its contents.

        For an on-disk cache, this reduces the cache file to the minimum possible size.
        """
        if self.cache_backend is not None:
            self.cache_backend.clear()
