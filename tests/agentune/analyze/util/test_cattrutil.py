from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from typing import override

import attrs
import cattrs
import pytest
from attrs import define, frozen

from agentune.analyze.util import cattrutil
from agentune.analyze.util.cattrutil import OverrideTypeTag, UseTypeTag

_logger = logging.getLogger(__name__)

def test_use_type_tag() -> None:

    # This hierarchy includes:
    # - generics
    # - abstract classes besides the base
    # - multiple inheritance diamonds
    
    @frozen
    class Base[T](ABC, UseTypeTag):
        @property
        @abstractmethod
        def j(self) -> T: ...

    @frozen
    class Sub1(Base[int]):
        j: int

    @frozen
    class Sub2(Base[int]):
        j: int # Same field, can't distinguish from Sub1 by field names

    @frozen
    class Sub3(Sub2): pass

    @frozen
    class Sub4(Sub2, OverrideTypeTag):

        @override
        @classmethod
        def _type_tag(cls) -> str:
            return 'Subby4'

    # Not decorated with @define; is abstract (does not implement a 'j') and cannot be instantiated
    class Sub5[T](Base[T]): pass

    @frozen
    class Sub6(Sub5[str]):
        j: str

    @frozen(slots=False)
    class Mid1[T](Base[T]):
        i: int

    @frozen(slots=False)
    class Mid2[T](Base[T]):
        b: bool

    @frozen(slots=False)
    class Sub7(Mid1[str], Mid2[str]):
        x: str

        @override
        @property
        def j(self) -> str:
            return self.x

    converter = cattrs.preconf.json.make_converter()
    cattrutil.register_use_type_tag(converter)

    assert converter.unstructure(Sub1(j=1), Base) == {'j': 1, '_type': 'Sub1'}
    assert converter.unstructure(Sub1(j=1)) == {'j': 1, '_type': 'Sub1'}
    assert converter.unstructure(Sub2(j=1), Base) == {'j': 1, '_type': 'Sub2'}
    assert converter.unstructure(Sub2(j=1)) == {'j': 1, '_type': 'Sub2'}
    assert converter.unstructure(Sub3(j=1), Base) == {'j': 1, '_type': 'Sub3'}
    assert converter.unstructure(Sub3(j=1)) == {'j': 1, '_type': 'Sub3'}
    assert converter.unstructure(Sub4(j=1), Base) == {'j': 1, '_type': 'Subby4'}
    assert converter.unstructure(Sub4(j=1)) == {'j': 1, '_type': 'Subby4'}
    assert converter.unstructure(Sub6(j='1'), Base) == {'j': '1', '_type': 'Sub6'}
    assert converter.unstructure(Sub6(j='1')) == {'j': '1', '_type': 'Sub6'}

    assert converter.loads(converter.dumps(Sub1(j=1)), Base) == Sub1(j=1) # type: ignore[type-abstract]
    assert converter.loads(converter.dumps(Sub2(j=1)), Base) == Sub2(j=1) # type: ignore[type-abstract]
    assert converter.loads(converter.dumps(Sub3(j=1)), Base) == Sub3(j=1) # type: ignore[type-abstract]
    assert converter.loads(converter.dumps(Sub4(j=1)), Base) == Sub4(j=1) # type: ignore[type-abstract]
    assert converter.loads(converter.dumps(Sub1(j=1)), Sub1) == Sub1(j=1)
    assert converter.loads(converter.dumps(Sub2(j=1)), Sub2) == Sub2(j=1)
    assert converter.loads(converter.dumps(Sub3(j=1)), Sub3) == Sub3(j=1)
    assert converter.loads(converter.dumps(Sub4(j=1)), Sub4) == Sub4(j=1)

    assert converter.loads(converter.dumps(Sub3(j=1)), Sub2) == Sub3(j=1), 'Structures subclass when structure_as is concrete'
    assert converter.loads(converter.dumps(Sub4(j=1)), Sub2) == Sub4(j=1), 'Structures subclass when structure_as is concrete'

    with pytest.raises(TypeError, match='abstract class'):
        Sub5() # type: ignore[abstract]

    with pytest.raises(ValueError, match='Unfamiliar type tag'):
        converter.structure({'j': 1, '_type': 'nonesuch'}, Base) # type: ignore[type-abstract]
    with pytest.raises(ValueError, match='Missing type tag'):
        converter.structure({'j': 1}, Base) # type: ignore[type-abstract]

    assert converter.loads(converter.dumps(Sub6(j='1')), Base) == Sub6(j='1') # type: ignore[type-abstract]
    assert converter.loads(converter.dumps(Sub6(j='1')), Sub6) == Sub6(j='1')

    assert converter.unstructure(Sub7(i=1, b=True, x='1')) == {'i': 1, 'b': True, 'x': '1', '_type': 'Sub7'}
    assert converter.loads(converter.dumps(Sub7(i=1, b=True, x='1')), Base) == Sub7(i=1, b=True, x='1') # type: ignore[type-abstract]
    assert converter.loads(converter.dumps(Sub7(i=1, b=True, x='1')), Sub7) == Sub7(i=1, b=True, x='1')
    assert converter.loads(converter.dumps(Sub7(i=1, b=True, x='1')), Mid1) == Sub7(i=1, b=True, x='1') # type: ignore[type-abstract]
    assert converter.loads(converter.dumps(Sub7(i=1, b=True, x='1')), Mid2) == Sub7(i=1, b=True, x='1') # type: ignore[type-abstract]

def test_parent_non_attrs() -> None:
    """Test when the abstract parent class is not itself an attrs class"""
    class Base[T](ABC, UseTypeTag):
        @property
        @abstractmethod
        def j(self) -> T: ...

    @frozen
    class Sub1(Base[int]):
        j: int

    @frozen
    class Sub2(Base[int]):
        j: int # Same field, can't distinguish from Sub1 by field names

    converter = cattrs.preconf.json.make_converter()
    cattrutil.register_use_type_tag(converter)

    assert converter.unstructure(Sub1(j=1), Base) == {'j': 1, '_type': 'Sub1'}
    assert converter.unstructure(Sub1(j=1)) == {'j': 1, '_type': 'Sub1'}
    assert converter.unstructure(Sub2(j=1), Base) == {'j': 1, '_type': 'Sub2'}
    assert converter.unstructure(Sub2(j=1)) == {'j': 1, '_type': 'Sub2'}

    assert converter.loads(converter.dumps(Sub1(j=1)), Base) == Sub1(j=1) # type: ignore[type-abstract]
    assert converter.loads(converter.dumps(Sub2(j=1)), Base) == Sub2(j=1) # type: ignore[type-abstract]

def test_add_subclass_after_registration() -> None:
    """Test defining (realistically, loading a module that defines) a subclass after the converter is created."""
    @define
    class Base(UseTypeTag):
        pass

    @frozen
    class Sub1(Base):
        j: int

    @frozen
    class Sub2(Base):
        j: int

    converter = cattrs.preconf.json.make_converter()
    cattrutil.register_use_type_tag(converter)

    @frozen
    class Sub3(Base):
        j: int

    assert converter.loads(converter.dumps(Sub1(j=1)), Base) == Sub1(j=1)
    assert converter.loads(converter.dumps(Sub2(j=1)), Base) == Sub2(j=1)
    assert converter.loads(converter.dumps(Sub3(j=1)), Base) == Sub3(j=1)

def test_delete_subclass_after_registration() -> None:
    """Test undefining (realistically, unloading a module that defines) a subclass after the converter is created."""
    @define
    class Base(UseTypeTag):
        pass

    @frozen
    class Sub1(Base):
        j: int

    @frozen
    class Sub2(Base):
        j: int

    converter = cattrs.preconf.json.make_converter()
    cattrutil.register_use_type_tag(converter)

    dumped = converter.dumps(Sub2(j=1))
    assert converter.loads(dumped, Base) == Sub2(j=1)

    del Sub2
    while gc.collect() > 0: pass

    _logger.info(converter.loads(dumped, Base))

def test_structure_without_unstructure() -> None:
    """Regression test for a bug that caused the UseTypeTag structure hook to fail if unstructure wasn't called before
    for that type.
    """
    @define
    class Base(UseTypeTag):
        name: str

    assert attrs.fields(Base).name.type == 'str' # Not the type str, the value 'str', because we import annotations from __future__

    converter = cattrs.Converter()
    cattrutil.register_use_type_tag(converter)

    assert converter.structure({'name': 'foo', '_type': 'Base'}, Base) == Base('foo')

