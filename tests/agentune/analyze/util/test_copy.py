from __future__ import annotations

import dataclasses
from typing import Any

import attrs

from agentune.analyze.util.copy import replace


def test_replace() -> None:
    @attrs.define
    class A:
        a: int
        b: str

    @dataclasses.dataclass
    class B:
        a: int
        b: str
    
    class C:
        def __init__(self, a: int, b: str) -> None:
            self.a = a
            self.b = b
        def __replace__(self, **kwargs: Any) -> C:
            return C(**{**self.__dict__, **kwargs})
        def __eq__(self, other: object) -> bool:
            return isinstance(other, C) and self.a == other.a and self.b == other.b
        def __str__(self) -> str:
            return f'C({self.a}, {self.b})'
    
    assert replace(A(1, 'b'), a=2) == A(2, 'b')
    assert replace(B(1, 'b'), a=2) == B(2, 'b')
    assert replace(C(1, 'b'), a=2) == C(2, 'b')

    
