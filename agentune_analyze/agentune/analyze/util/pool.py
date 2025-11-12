"""Thread-safe pool implementation for managing reusable instances of arbitrary type T."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from datetime import timedelta
from typing import Protocol, Self

from attrs import field, frozen
from janus import SyncQueueEmpty, SyncQueueShutDown

from agentune.analyze.util.queue import Queue

_logger = logging.getLogger(__name__)


class PoolTimeoutError(Exception):
    """Raised when acquiring from pool times out."""


class PoolClosedError(Exception):
    """Raised when attempting to acquire from a closed pool."""


class Closable(Protocol):
    """Protocol for objects that can be closed."""
    
    def close(self) -> None:
        """Close the resource."""
        ...


@frozen(eq=False, hash=False)
class ThreadsafePool[T: Closable]:
    """A thread-safe pool for managing instances of type T, which must have a close() method.
    
    The pool immediately creates `size` instances; if this fails, the pool's constructor raises that error,
    after close()ing any instances it succeeded in creating.

    If a call to acquire() raises an error inside the context, that instance of T is close()d and a new one is created instead,
    and the original error is reraised.
    However, if that call to factory() fails, the pool is automatically closed, and that exception is raised instead.

    The pool never shrinks.

    When the pool itself is close()d, all blocking threads are released (by raising PoolClosedError),
    all instances in the pool are T.close()d, and all instances returned to the pool later are close()d then.

    Errors raised by T.close() are ignored and not propagated except in the pool constructor.
    """
    
    factory: Callable[[], T]
    size: int
    acquire_timeout: timedelta
    # We don't pass size to the queue, we manage it ourselves; we never need to block on the queue when *inserting*.
    # We use our Queue for the shutdown() support, which was only added to stdlib queue.Queue in python 3.13.
    _pool: Queue[T] = field(init=False)
    @_pool.default
    def _pool_init(self) -> Queue[T]:
        queue: Queue[T] = Queue(self.size)
        try:
            for _ in range(self.size):
                queue.put(self.factory())
            return queue
        except:
            queue.close()
            for instance in queue:
                self._close_instance(instance)
            raise

    def _close_instance(self, instance: T) -> None:
        with suppress(Exception):
            instance.close()

    @contextmanager
    def acquire(self, timeout: timedelta | None = None) -> Iterator[T]:
        """Acquire an instance from the pool. Must be used as a context manager to return it to the pool.

        If the caller raises an error inside the context exits with an error, that instance is close()d and not returned
        to future callers.
        
        Args:
            timeout: How long to wait for an available instance. Defaults to the pool's acquire_timeout.

        Raises:
            PoolTimeoutError: If timeout expires before instance becomes available.
            PoolClosedError: If pool has been closed.
        """
        if timeout is None:
            timeout = self.acquire_timeout

        timeout_seconds = timeout.total_seconds() if timeout is not None else None

        try:
            instance = self._pool.get(timeout_seconds)
        except SyncQueueEmpty:
            raise PoolTimeoutError(f'Timeout after {timeout} waiting for pool instance') from None
        except SyncQueueShutDown:
            raise PoolClosedError from None

        try:
            yield instance
        except:
            self._close_instance(instance)
            try:
                new_instance = self.factory()
            except:
                _logger.exception('Error while creating instance to replace previously failed instance, shutting down the pool')
                self.close()
                raise
            try:
                self._pool.put(new_instance)
            except SyncQueueShutDown:
                # Race between us and closing the pool
                self._close_instance(instance)

            raise # Propagate the exception raised by the caller of acquire()
        else:
            try:
                self._pool.put_nowait(instance)
            except SyncQueueShutDown:
                # Race between us and closing the pool
                self._close_instance(instance)


    def close(self) -> None:
        """Close all instances in the pool. No more instances can be acquired. Instances returned to the pool will be closed."""
        self._pool.close()

        try:
            while True:
                instance = self._pool.get_nowait()
                if instance is not None:
                    with suppress(Exception):
                        instance.close()
                else: # Should not happen when the queue has shut down
                    break
        except SyncQueueShutDown:
            pass

    def __enter__(self) -> Self:
        """Call close() on exiting the context."""
        return self
    
    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
