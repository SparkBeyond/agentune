from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from typing import overload, override

from attrs import field, frozen


@frozen
class LRUCache[K, V](MutableMapping[K, V]):
    """A simple LRU cache exposed as a mapping."""

    maxsize: int
    _cache: OrderedDict[K, V] = field(init=False, factory=OrderedDict)

    @override
    def __getitem__(self, key: K) -> V:
        if key not in self._cache:
            raise KeyError(key)
        self._cache.move_to_end(key)
        return self._cache[key]

    @override
    def __setitem__(self, key: K, value: V) -> None:
        if key in self._cache:
            self._cache[key] = value
            self._cache.move_to_end(key)
        else:
            self._cache[key] = value
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    @override
    def __delitem__(self, key: K) -> None:
        del self._cache[key]

    @override
    def __iter__(self) -> Iterator[K]:
        return iter(self._cache)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: object) -> bool:
        return key in self._cache

    # Have to redeclare all overloads when overriding or mypy complains

    @overload
    def get(self, key: K) -> V | None: ...

    @overload
    def get[T](self, key: K, default: T) -> V | T: ...

    @override
    def get[T](self, key: K, default: V | T | None = None) -> V | T | None:
        try:
            return self[key]
        except KeyError:
            return default

    @override
    def clear(self) -> None:
        self._cache.clear()
