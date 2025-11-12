import asyncio
from abc import abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any, Protocol, Self

from attrs import field, frozen


class KVStore[K, V](Protocol):
    """A subset of MutableMapping that doesn't support iteration, fetching the stored keys, or clear()."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, key: K) -> V: ...

    def get(self, key: K) -> V | None:
        try:
            return self[key]
        except KeyError:
            return None

    def __contains__(self, key: K) -> bool:
        return self.get(key) is not None

    @abstractmethod
    def __setitem__(self, key: K, value: V) -> None: ...

    @abstractmethod
    def __delitem__(self, key: K) -> None: ...


@frozen(eq=False, hash=False)
class HalfAsyncMap[K, V]:
    """A mutable mapping with atomic get-or-update semantics for asynchronous producer functions.

    Once a value is known, it is stored (and can be modified or deleted) synchronously,
    hence this is called a 'half async' map.

    Calling get_or_update fetches an existing value from storage, if one exists;
    otherwise, awaits the results of a previous call to the async producer function, if one is outstanding for this key;
    otherwise, calls the async function.

    The atomicity guarantee is that an async producer function will never be called for a given key if
    a previous call for the same key has not yet returned.

    Supports the underlying storage spontaneously removing entries, e.g. because it enforces a maximum size,
    so it can be used with e.g. LRUCache.
    """
    _storage: KVStore[K, V] = field(factory=dict, kw_only=True, alias='storage')
    _outstanding: dict[K, asyncio.Task[V]] = field(init=False, factory=dict)

    async def get_or_update(self, key: K, producer: Callable[[], Coroutine[Any, Any, V]]) -> V:
        stored = self._storage.get(key)
        if stored is not None:
            return stored
        task = self._outstanding.get(key)
        if task is not None:
            return await task
        coroutine = producer()
        task = asyncio.create_task(coroutine)
        self._outstanding[key] = task
        result = await task
        self._storage[key] = result
        del self._outstanding[key]
        return result

    async def join_all_outstanding(self) -> None:
        """Wait for all currently outstanding calls to the producer function to complete.

        Ignores calls started after the call to this method.
        Does not propagate errors if any currently outstanding calls fail.
        """
        await asyncio.gather(*list(self._outstanding.values()), return_exceptions=True)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.join_all_outstanding()


@frozen(eq=False, hash=False)
class LoadingHalfAsyncMap[K, V](HalfAsyncMap[K, V]):
    """As AsyncMap, with a default producer K => Coroutine[Any, Any, V] which can be used in get_or_update."""
    producer: Callable[[K], Coroutine[Any, Any, V]]

    async def simple_get_or_update(self, key: K) -> V:
        async def producer_wrapper() -> V:
            return await self.producer(key)
        return await self.get_or_update(key, producer_wrapper)

