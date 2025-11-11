from collections.abc import Callable
from typing import Any

import cattrs
import cattrs.preconf.json
import httpx
from attrs import field, frozen
from cattrs import Converter
from cattrs.preconf.json import JsonConverter
from llama_index.core.llms import LLM

from agentune.analyze.core.llm import LLMContext, LLMSpec
from agentune.analyze.util import cattrutil

# Custom hooks that don't need a SerializationContext should be registered directly with this converter.
# Do NOT use this converter to actually convert values, because some other hooks are registered when a
# SerializationContext is created. Only ever use SerializationContext.converter to convert values.
converter_for_hook_registration: JsonConverter = cattrs.preconf.json.make_converter()

@frozen(eq=False, hash=False)
class SerializationContext:
    """Provides all live values that may be required when deserializing any class defined in this library.

    An instance of this class makes a copy of `register_context_converter_hook`; hooks registered with the converter afterward
    will not affect existing instances of SerializationContext. Some modules may need to be imported (and so given
    the chance to register hooks) before creating an instance.
    """

    llm_context: LLMContext
    # We may add DuckdbManager here, and other future fixtures. Or make another class containing both this and them.

    # The final converter to be used, containing all hooks (context-aware and non-context-aware) that have been registered 
    # with this module when this context was created.
    converter: JsonConverter = field(init=False)
    @converter.default
    def _converter_default(self) -> JsonConverter:
        conv = converter_for_hook_registration.copy()
        _register_context_aware_functions(self, conv)
        return conv

    @property
    def httpx_client(self) -> httpx.Client:
        return self.llm_context.httpx_client
    
    @property
    def httpx_async_client(self) -> httpx.AsyncClient:
        return self.llm_context.httpx_async_client

# This class is in this module to avoid circular imports

@frozen
class LLMWithSpec:
    """Classes that use LLMs should use this as the parameter type.

    It is unstructured by storing only the LLMSpec, and automatically structured
    as an LLMWithSpec in the presence of a SerializationContext.

    Equality ignores the LLM and compares only the LLMSpec. (Instances of class LLM are not hashable and don't have
    well-defined equality.)
    """
    spec: LLMSpec
    llm: LLM

    def __hash__(self) -> int:
        return hash(self.spec)

    def __eq__(self, value: object) -> bool:
        return isinstance(value, LLMWithSpec) and self.spec == value.spec

    # LLM instances must not be stringified, they leak keys and other secrets
    def __str__(self) -> str:
        return f'LLMWithSpec({self.spec}, <{type(self.llm).__name__}>)'

    def __repr__(self) -> str:
        return str(self)

# Variants on cattrs register/unregister functions that additionally use a SerializationContext.

type ContextConverterHook = Callable[[SerializationContext, cattrs.Converter], None]

_context_converter_hooks: tuple[ContextConverterHook, ...] = ()

def register_context_converter_hook(func: ContextConverterHook) -> ContextConverterHook:
    global _context_converter_hooks # noqa: PLW0603
    _context_converter_hooks = (*_context_converter_hooks, func)
    return func

def _register_context_aware_functions(context: SerializationContext, converter: cattrs.Converter) -> None:
    # A tuple is safe to read without holding the lock
    for converter_hook in _context_converter_hooks:
        converter_hook(context, converter)

    # Apply this last so it can capture custom hooks registered for other types
    cattrutil.register_use_type_tag(converter)

@register_context_converter_hook
def _structure_llm(context: SerializationContext, converter: Converter) -> None:
    prev_structure = converter.get_structure_hook(LLMSpec)
    prev_unstructure = converter.get_unstructure_hook(LLMSpec)

    @converter.register_structure_hook
    def structure(value: Any, _cl: type) -> LLMWithSpec:
        spec = prev_structure(value, LLMSpec)
        return LLMWithSpec(spec, context.llm_context.from_spec(spec))

    @converter.register_unstructure_hook
    def unstructure(value: LLMWithSpec) -> Any:
        return prev_unstructure(value.spec)

