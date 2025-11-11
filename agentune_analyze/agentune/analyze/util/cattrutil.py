from __future__ import annotations

import logging
import threading
import typing
import weakref
from abc import ABC, abstractmethod
from typing import Any

import attrs
from cattrs import Converter
from cattrs.dispatch import StructureHook, UnstructureHook

_logger = logging.getLogger(__name__)


class UseTypeTag:
    """Marker base class for serializable class hierarchies.

    Subclasses are unstructured with a `_type` field containing the class name.

    Structuring will take into account the subclasses defined at that time; it cannot locate and import the right module
    that defines a not-yet-loaded subclass.

    Usage:
        @attrs.define
        class Parent(SerializeHierarchy):
            pass

        @attrs.define
        class Child1(Parent):
            field1: str

        @attrs.define
        class Child2(Parent):
            field2: int

        # Child1 instances serialize as: {"_type": "Child1", "field1": "value"}
        # Child2 instances serialize as: {"_type": "Child2", "field2": 42}
    """
    '''Marker class which enables the include_subclasses_by_name strategy for its descendants.

    Descendants will be un/structured as dicts with an extra field '_type' containing the result of `_type_tag()`,
    by default the class name.
    
    '''
    @classmethod
    def _type_tag(cls) -> str:
        return cls.__name__

    # class vars
    _subclass_by_tag_cache_lock = threading.Lock()
    _subclass_by_tag_cache = weakref.WeakKeyDictionary[type['UseTypeTag'], weakref.WeakValueDictionary[str, type['UseTypeTag']]]()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        with UseTypeTag._subclass_by_tag_cache_lock:
            UseTypeTag._subclass_by_tag_cache.clear()
        super().__init_subclass__(**kwargs)


def register_use_type_tag(converter: Converter, tag_name: str = '_type') -> None:
    def unstructure_factory(_cl: type[UseTypeTag], converter: Converter) -> UnstructureHook:
        unstructure_hooks: dict[type, UnstructureHook] = {}

        def unstructure_hook(obj: UseTypeTag) -> dict[str, Any]:
            real_cl = type(obj)
            prev_hook = unstructure_hooks.get(real_cl)
            if prev_hook is None:
                prev_hook = converter.gen_unstructure_attrs_fromdict(real_cl)
                unstructure_hooks[real_cl] = prev_hook
            dct = prev_hook(obj)
            dct[tag_name] = obj._type_tag()
            return dct
        return unstructure_hook

    def check_subclass(cl: type) -> bool:
        return issubclass(typing.get_origin(cl) or cl, UseTypeTag)

    converter.register_unstructure_hook_factory(check_subclass, unstructure_factory)

    def structure_factory(cl: type[UseTypeTag], converter: Converter) -> StructureHook:
        def structure_hook(dct: dict[str, Any], target_type: type[UseTypeTag]) -> UseTypeTag:
            tag = dct.pop(tag_name, None)
            if tag is None:
                raise ValueError(f'Missing type tag field {tag_name} to structure {target_type.__name__}')
            if not isinstance(tag, str):
                raise ValueError(f'{tag_name} type tag field must be a string but found {type(tag)}: {tag}')

            with UseTypeTag._subclass_by_tag_cache_lock:
                class_by_tag = UseTypeTag._subclass_by_tag_cache.get(cl)
                if class_by_tag is None:
                    subclasses = [typing.get_origin(sub) or sub for sub in _make_subclasses_tree(cl)]
                    concrete_classes = [cl for cl in subclasses if not _is_abstract(cl)]
                    class_by_tag = weakref.WeakValueDictionary(_deduplicate_class_by_tag(concrete_classes))
                    UseTypeTag._subclass_by_tag_cache[cl] = class_by_tag

            real_cl = class_by_tag.get(tag)
            if real_cl is None:
                raise ValueError(f'Unfamiliar type tag value {tag} for subtype of {target_type.__name__}; '
                                 f'did you forget to import a module?')

            # Work around python-attrs/cattrs#692
            if attrs.has(real_cl):
                attrs.resolve_types(real_cl)

            # Can't use converter.gen_structure_attrs_fromdict here because it doesn't handle generic classes like JoinStrategy.
            # (These are classes where the type param doesn't affect any attrs fields, only method signatures.)
            # Because of this we can't cache the previous hook; this may slightly hurt performance.
            return typing.cast(UseTypeTag, converter.structure_attrs_fromdict(dct, real_cl))

        return structure_hook

    converter.register_structure_hook_factory(check_subclass, structure_factory)



class OverrideTypeTag(ABC):
    """Extend this class to override the type tag (discriminator value) used when de/structuring instances.

    This affects lazy_include_subclasses_by_name(), not any of the other methods in this module or any standard
    cattr strategies.

    It applies to any class that extends this class, so you have to either return the right value for your subclasses
    or override this method again in every sublass.
    """
    @classmethod
    @abstractmethod
    def _type_tag(cls) -> str: ...

def _tag_generator(cl: type) -> str:
    real_cl: type = typing.get_origin(cl) or cl # If cl is Foo[T] or even Foo[str] we need it to be plain Foo
    if issubclass(real_cl, OverrideTypeTag):
        return real_cl._type_tag()
    return real_cl.__name__

# Copy of private functions from cattrs.strategies
# with set() applied to fix python-attrs/cattrs#682
def _make_subclasses_tree[T](cl: type[T]) -> list[type[T]]:
    cls_origin = typing.get_origin(cl) or cl
    return list(set([cl] + [
        sscl
        for scl in cls_origin.__subclasses__()
        for sscl in _make_subclasses_tree(scl)
    ]))

def _is_abstract(cl: type) -> bool:
    if not issubclass(cl, ABC):
        return False
    return len(cl.__abstractmethods__) > 0

def _deduplicate_class_by_tag(classes: list[type]) -> dict[str, type]:
    """Match pairs of original+attrs classes and remove the original class; return a mapping of type tags to classes.

    When attrs creates a slots class, it replaces the original class, but both classes exist in memory
    (until GC) and in __subclasses__. We want to match such pairs and return from each pair only the final (attrs-created)
    class, and also return any unmatched classes (which might be non-attrs or non-slots classes).

    The matching is done naively, by name; this is not correct in the general case but it's good enough
    for our use case since we use the name to identify classes anyway.
    """
    # attrs.has returns True for subclasses of an (unrelated) parent attrs class; the initial subclass is already an attrs class
    # and not only the final attrs-processed subclass.
    # To distinguish the two, we need to check if __attrs_attrs__ is defined on this exact class and not some parent.
    def is_really_attrs_class(cl: type) -> bool:
        return '__attrs_attrs__' in vars(cl)

    result: dict[str, type] = {}
    for cl in classes:
        tag = _tag_generator(cl)
        if tag in result:
            existing = result[tag]
            if existing is not cl:
                if is_really_attrs_class(existing) and not is_really_attrs_class(cl):
                    pass
                elif is_really_attrs_class(cl) and not is_really_attrs_class(existing):
                    result[tag] = cl
                else:
                    raise ValueError(f'Duplicate type tag {tag} for classes {cl} and {existing}')
        else:
            result[tag] = cl
    return result
