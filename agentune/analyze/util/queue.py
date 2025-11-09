import logging
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable, Iterator

import janus
from attrs import field, frozen
from janus import AsyncQueueShutDown, SyncQueueEmpty, SyncQueueShutDown

_logger = logging.getLogger(__name__)

class Queue[T](Iterable[T], AsyncIterable[T]):
    """Queue connecting sync and async. Also supports for and async for, which wait for the queue to be closed.

    This wraps a janus.Queue (which has the same interfaces as python normal and asyncio Queues), adds some
    high-level methods and deliberately does not expose some other functionality; the semantics are slightly
    different from the builtin and janus queues:

    - There is no task_done() method. Whenever an item is removed from the queue (e.g. with `get` or `produce`),
      task_done() is automatically called on the internal queue.
    - There is no join() method (which requires task_done semantics). Instead, there is a wait_complete() method,
      which returns as soon as the queue is empty.
    """
    def __init__(self, maxsize: int = 1):
        self.maxsize = maxsize
        self._queue: janus.Queue[T] = janus.Queue(maxsize)

    def close(self, immediate: bool = False) -> None:
        """After this call, attempts to put more items will raise an error.
        
        If immediate is False, attempts to get items will succeed, and start raising errors when the queue is empty.
        If immediate is True, attempts to get items will raise errors immediately, and the items in the queue will be discarded.
        """
        self._queue.shutdown(immediate=immediate)

    def put(self, item: T, timeout: float | None = None) -> None:
        if timeout == 0.0:
            self._queue.sync_q.put_nowait(item) # More efficient codepath
        else:
            self._queue.sync_q.put(item, timeout=timeout)

    def put_nowait(self, item: T) -> None:
        self._queue.sync_q.put_nowait(item)

    def get(self, timeout: float | None = None) -> T:
        if timeout == 0.0:
            ret = self._queue.sync_q.get_nowait() # More efficient codepath
        else:
            ret = self._queue.sync_q.get(timeout=timeout)
        self._queue.sync_q.task_done()
        return ret
    
    def get_nowait(self) -> T | None:
        try:
            ret = self._queue.sync_q.get_nowait()
            self._queue.sync_q.task_done()
            return ret
        except SyncQueueEmpty:
            return None
    
    def get_batch(self, n: int) -> list[T]:
        ret = []
        try:
            for _ in range(n):
                ret.append(self.get())
        except SyncQueueShutDown:
            pass
        return ret
    
    def get_batches(self, n: int) -> Iterator[list[T]]:
        while True:
            ret = []
            try:
                for _ in range(n):
                    ret.append(self.get())
                yield ret
            except SyncQueueShutDown:
                yield ret
                raise StopIteration from None

    async def aput(self, item: T) -> None:
        await self._queue.async_q.put(item)

    async def aget(self) -> T:
        ret = await self._queue.async_q.get()
        self._queue.async_q.task_done()
        return ret
    
    async def aget_batch(self, n: int) -> list[T]:
        ret = []
        try:
            for _ in range(n):
                ret.append(await self.aget())
        except AsyncQueueShutDown:
            pass
        return ret
    
    async def aget_batches(self, n: int) -> AsyncIterator[list[T]]:
        while True:
            ret = []
            try:
                for _ in range(n):
                    ret.append(await self.aget())
                yield ret
            except AsyncQueueShutDown:
                yield ret
                raise StopAsyncIteration from None
    
    def consume(self, iterable: Iterable[T], then_close: bool = True) -> None:
        for x in iterable:
            self.put(x)
        if then_close:
            self._queue.shutdown()

    async def aconsume(self, iterable: AsyncIterable[T], then_close: bool = True) -> None:
        async for x in iterable:
            await self.aput(x)
        if then_close:
            self._queue.shutdown()

    def produce(self, into: Callable[[T], None]) -> None:
        for x in self:
            into(x)

    async def aproduce(self, into: Callable[[T], Awaitable[None]]) -> None:
        async for x in self:
            await into(x)

    def wait_empty(self) -> None:
        """Wait for the queue to be empty.

        Does not close the queue; items can be added again later, and then you call wait_empty again.

        Note that the semantics are different from python Queue.join(); this returns as soon as the queue is empty,
        without a need for you to call task_done().
        """
        self._queue.sync_q.join()

    async def await_empty(self) -> None:
        await self._queue.async_q.join()

    def __iter__(self) -> Iterator[T]:
        try:
            while True:
                yield self.get()
        except SyncQueueShutDown:
            pass

    async def __aiter__(self) -> AsyncIterator[T]:
        try:
            while True:
                yield await self.aget()
        except AsyncQueueShutDown:
            pass
        
    def __len__(self) -> int:
        return self._queue.sync_q.qsize()


# High level methods

# Can't use contextlib.contextmanager, it doesn't support generics
@frozen
class ScopedQueue[T]:
    """Context manager for Queue[T]. Closes it when going out of scope and logs a warning if it is not empty at that point."""
    maxsize: int = 1
    queue: Queue[T] = field(init=False)
    @queue.default
    def _queue(self) -> Queue[T]:
        return Queue(maxsize=self.maxsize)
    
    def __enter__(self) -> Queue[T]:
        return self.queue

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self.queue._queue.sync_q.qsize() > 0:
            _logger.warning(f'Non-empty ScopedQueue going out of scope, {exc_val=}')
        self.queue.close()

    async def __aenter__(self) -> Queue[T]:
        return self.queue
    
    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self.queue._queue.async_q.qsize() > 0:
            _logger.warning(f'Non-empty ScopedQueue going out of scope, {exc_val=}')
        self.queue.close()

