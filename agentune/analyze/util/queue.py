import asyncio
import contextlib
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator

import janus
from janus import AsyncQueueShutDown, SyncQueueEmpty, SyncQueueShutDown


# Queue connecting sync and async. Also supports for and async for, which wait for the queue to be closed.
# Future multiprocess work should also implement this interface.
class Queue[T](Iterable[T], AsyncIterable[T]):
    def __init__(self, maxsize: int = 1):
        self.maxsize = maxsize
        self._queue: janus.Queue[T] = janus.Queue(maxsize)

    def close(self, immediate: bool = False) -> None:
        """After this call, attempts to put more items will raise an error.
        
        If immediate is False, attempts to get items will succeed, and start raising errors when the queue is empty.
        If immediate is True, attempts to get items will raise errors immediately, and the items in the queue will be discarded.
        """
        self._queue.shutdown(immediate=immediate)

    def put(self, item: T) -> None:
        self._queue.sync_q.put(item)

    def get(self) -> T:
        return self._queue.sync_q.get()
    
    def get_nowait(self) -> T | None:
        try:
            return self._queue.sync_q.get_nowait()
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
        return await self._queue.async_q.get()
    
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
    
    def consume(self, iterator: Iterator[T], then_close: bool = True) -> None:
        for x in iterator:
            self.put(x)
        if then_close:
            self._queue.shutdown()

    async def aconsume(self, iterator: AsyncIterator[T], then_close: bool = True) -> None:
        async for x in iterator:
            await self.aput(x)
        if then_close:
            self._queue.shutdown()

    # Wait for queue to empty. Does not close the queue.
    def wait_empty(self) -> None:
        self._queue.sync_q.join()

    # Wait for queue to empty. Does not close the queue.
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

@contextlib.contextmanager
def scoped_queue[T](maxsize: int = 1) -> Iterator[Queue[T]]:
    queue = Queue[T](maxsize)
    try:
        yield queue
    finally:
        queue.close()
        queue.wait_empty()

@contextlib.asynccontextmanager
async def ascoped_queue[T](maxsize: int = 1) -> AsyncIterator[Queue[T]]:
    queue = Queue[T](maxsize)
    try:
        yield queue
    finally:
        queue.close()
        await queue.await_empty()

def sync_to_async_iter[T](iter: Iterator[T]) -> AsyncIterator[T]:
    """Convert a sync iterator to an async iterator, scheduling the blocking calls to the sync iterator on a new thread.
    
    This method is defined as async to make it clear it should only be called from an async context.
    """
    queue = Queue[T]()
    _ = asyncio.to_thread(queue.consume, iter)
    return aiter(queue)
