from abc import ABC

import cattrs
import cattrs.dispatch
import cattrs.preconf.json
import cattrs.strategies
from attrs import frozen

from agentune.analyze.util import cattrutil


def test_dynamic_include_subclasses() -> None:
    
    converter = cattrs.Converter()
    
    @cattrutil.lazy_include_subclasses_by_name(converter)
    @frozen
    class Base(ABC): 
        pass
    
    @frozen 
    class Sub1(Base):
        j: int

    @frozen 
    class Sub2(Base):
        j: int # Same field, can't distinguish from Sub1

    assert converter.unstructure(Sub1(j=1), Base) == {'j': 1, 'type_tag': 'Sub1'}
    assert converter.unstructure(Sub1(j=1)) == {'j': 1, 'type_tag': 'Sub1'}
    assert converter.unstructure(Sub2(j=1), Base) == {'j': 1, 'type_tag': 'Sub2'}
    assert converter.unstructure(Sub2(j=1)) == {'j': 1, 'type_tag': 'Sub2'}

    assert converter.structure(converter.unstructure(Sub1(j=1)), Base) == Sub1(j=1)
    assert converter.structure(converter.unstructure(Sub2(j=1)), Base) == Sub2(j=1)
    assert converter.structure(converter.unstructure(Sub1(j=1)), Sub1) == Sub1(j=1)
    assert converter.structure(converter.unstructure(Sub2(j=1)), Sub2) == Sub2(j=1)


if __name__ == '__main__':
    test_dynamic_include_subclasses()
