from attrs import frozen
from llama_index.core.llms import LLM

from agentune.analyze.api.base import RunContext
from agentune.analyze.core.llm import LLMSpec
from agentune.analyze.core.sercontext import LLMWithSpec


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

