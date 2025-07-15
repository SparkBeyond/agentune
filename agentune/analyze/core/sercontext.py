import inspect
from collections.abc import Callable
from functools import partial
from typing import Any

import cattrs
import cattrs.dispatch
import cattrs.preconf.json
import cattrs.strategies
import httpx
from attrs import field, frozen
from llama_index.core.llms import LLM

from agentune.analyze.core.llm import LLMContext, LLMSpec
from agentune.analyze.core.serialize import default_converter


@frozen(eq=False, hash=False)
class SerializationContext:
    """Provides all live values that may be required when deserializing any class defined in this library."""

    llm: LLMContext
    # We may add DuckdbManager here, and other future fixtures. Or make another class containing both this and them.

    # The final converter to be used, containing all hooks (context-aware and non-context-aware) that have been registered 
    # with this module when this context was created.
    # TODO the next design step is to wrap the registries in a class and have a (resettable) instance of the class at the module level,
    # instead of multiple mutable module level variables. However, all the hooks we have and expect to have right now are ones
    # that can be registered globally and are always valid.
    converter: cattrs.Converter = field(init=False)
    @converter.default
    def _converter_default(self) -> cattrs.Converter:
        conv = default_converter.copy()
        _register_context_aware_functions(self, conv)
        return conv

    @property
    def httpx_client(self) -> httpx.Client:
        return self.llm.httpx_client
    
    @property
    def httpx_async_client(self) -> httpx.AsyncClient:
        return self.llm.httpx_async_client
    

# Variants on cattrs register/unregister functions that additionally use a SerializationContext.
# TODO add support for hook factories; add support for non-decorator use (as in plain cattrs).

type ContextStructureHook = Callable[[SerializationContext, Any, type], Any]
type ContextUnstructureHook = Callable[[SerializationContext, Any], Any]

_context_structure_hooks: list[ContextStructureHook] = []
_context_unstructure_hooks: list[ContextUnstructureHook] = []

def register_context_structure_hook(func: ContextStructureHook) -> ContextStructureHook:
    _context_structure_hooks.append(func)
    return func

def register_context_unstructure_hook(func: ContextUnstructureHook) -> ContextUnstructureHook:
    _context_unstructure_hooks.append(func)
    return func

def _register_context_aware_functions(context: SerializationContext, converter: cattrs.Converter) -> None:
    for structure_hook in _context_structure_hooks:
        return_type = inspect.signature(structure_hook).return_annotation
        converter.register_structure_hook(return_type, partial(structure_hook, context))
    for unstructure_hook in _context_unstructure_hooks:
        arg = list(inspect.signature(unstructure_hook).parameters.values())[1]
        arg_type = arg.annotation
        converter.register_unstructure_hook(arg_type, partial(unstructure_hook, context))

# This class is in this module to avoid circular imports

@frozen
class LLMWithSpec:
    """Classes that use LLMs should use this as the parameter type. It is unstructured by storing 
    only the LLMSpec, and automatically structured as an LLMWithSpec in the presence of a SerializationContext.
    """
    spec: LLMSpec
    llm: LLM


@register_context_structure_hook
def _structure_llm(context: SerializationContext, spec: LLMSpec, _: type) -> LLMWithSpec:
    return LLMWithSpec(spec, context.llm.from_spec(spec))

@default_converter.register_unstructure_hook
def _unstructure_llm(llm: LLMWithSpec) -> LLMSpec:
    return llm.spec
