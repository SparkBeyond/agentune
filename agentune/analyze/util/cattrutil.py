import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import cattrs
import cattrs.dispatch
import cattrs.strategies
from cattr import Converter


# From cattrs.strategies._subclasses; copied here because it's private and small
def _make_subclasses_tree(cl: type) -> list[type]:
    cls_origin = typing.get_origin(cl) or cl
    return [cl] + [
        sscl
        for scl in cls_origin.__subclasses__()
        for sscl in _make_subclasses_tree(scl)
    ]


def configure_lazy_include_subclasses[T](base: type[T], converter: cattrs.Converter, 
                                         tag_generator: Callable[[type[T]], str], tag_name: str) -> None:
    """Like cattrs.strategies.include_subclasses(), but takes effect on the first un/structure operation, 
    allowing subclasses to be loaded (and added to the parent's __subclasses__) first, including in other modules.

    The parameters and semantics are otherwise the same as for include_subclasses().

    This can be extended to support the loading and unloading of subclasses even after the first un/structure operation,
    although that would be a little slower.
    """
    def tag_generator_wrapper(cl: type[T]) -> str:
        if issubclass(cl, OverrideTypeTag):
            return cl.type_tag()
        return tag_generator(cl)

    # Can't wrap include_subclasses() in a hook factory, that results in an infinite recursion, so we roll our own.
    # This is much simpler than the implementation of include_subclasses() because we don't support the overrides parameter.

    # Only configure once, in structure or unstructure hook, whichever is called first. 
    # This is more efficient than @cache.
    cached_tag_to_type: dict[str, type[T]] | None = None
    cached_tag_to_unstructure_hook: dict[str, Callable[[T], dict[str, Any]]] | None = None
    
    def tag_to_type() -> dict[str, type[T]]:
        nonlocal cached_tag_to_type
        if cached_tag_to_type is None:
            cached_tag_to_type = {tag_generator_wrapper(cl): cl for cl in _make_subclasses_tree(base)}
        return cached_tag_to_type # TODO replace with @cached?
    
    def tag_to_unstructure_hook() -> dict[str, Callable[[T], dict[str, Any]]]:
        nonlocal cached_tag_to_unstructure_hook
        if cached_tag_to_unstructure_hook is None:
            cached_tag_to_unstructure_hook = {
                tag_generator_wrapper(cl): converter.gen_unstructure_attrs_fromdict(cl) # Despite name, works for non-attrs classes too
                for cl in _make_subclasses_tree(base) if cl is not base}
        return cached_tag_to_unstructure_hook # replace with @cached?

    # Structure hook registered only for Base itself; explicitly structuring a subclass works with the default behavior,
    # which means it does not require a type tag and ignores it if it is present, even if the tag has the wrong value.
    @converter.register_structure_hook_factory(lambda c: c is base)
    def _structure_hook(_: type) -> cattrs.dispatch.StructureHook:
        def hook(val: dict[str, Any], cl: type) -> T:
            try:
                tag = val[tag_name]
            except KeyError as e:
                raise KeyError(f'Type tag {tag_name} not found in {val}') from e
            try:
                subtype = tag_to_type()[tag]
            except KeyError as e:
                raise KeyError(f'Unexpected type tag {tag} for subclass of {cl}') from e
            return converter.structure(val, subtype)
        return hook
    
    # Unstructure hook registered for Base and all subclasses; even when expicitly unstructuring a subclass we want to write the tag.
    @converter.register_unstructure_hook_factory(lambda c: issubclass(c, base)) 
    def _unstructure_hook(_target_type: type[T], _base_converter: Converter) -> cattrs.dispatch.UnstructureHook:
        def hook(val: T) -> dict[str, Any]:
            tag = tag_generator_wrapper(type(val))
            base_hook = tag_to_unstructure_hook()[tag]
            base_dict = base_hook(val)
            if tag_name in base_dict and base_dict[tag_name] != tag:
                raise ValueError(f'Subclass of {_target_type} tagged {tag} already has attribute named {tag_name} with value {base_dict[tag_name]}')
            return base_hook(val) | {tag_name: tag}
        return hook

def lazy_include_subclasses[T](converter: cattrs.Converter, tag_generator: Callable[[type[T]], str],
                               tag_name: str = 'type_tag') -> Callable[[type[T]], type[T]]:
    def decorator(base: type[T]) -> type[T]:
        configure_lazy_include_subclasses(base, converter, tag_generator, tag_name)
        return base
    return decorator

type SameTypeConverter[T] = Callable[[type[T]], type[T]]


class OverrideTypeTag(ABC):
    """Extend this class to override the type tag (discriminator value) used when de/structuring instances.

    This affects lazy_include_subclasses_by_name(), not any of the other methods in this module.

    It applies to any class that extends this class, so you should return the right value for your subclasses
    (unless they override this method again).
    """
    @classmethod
    @abstractmethod
    def type_tag(cls) -> str: ...

def lazy_include_subclasses_by_name(converter: cattrs.Converter, 
                                    tag_name: str = 'type_tag') -> SameTypeConverter:
    def tag_generator(cl: type) -> str:
        if issubclass(cl, OverrideTypeTag):
            return cl.type_tag()
        return cl.__name__
    return lazy_include_subclasses(converter, tag_generator, tag_name)

