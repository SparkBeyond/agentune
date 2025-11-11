"""Shared (per context) limits on how many async features are computed at once."""
import asyncio
from asyncio import BoundedSemaphore
from collections.abc import Callable, Coroutine, Generator, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal, overload

from agentune.analyze.util import asynclimits

async_features_compute_semaphore = ContextVar[BoundedSemaphore | None]('async features eval limit', default=None)


def async_features_compute_limit() -> int | None:
    """Return the current limit as a number."""
    semaphore = async_features_compute_semaphore.get()
    if semaphore is not None:
        return asynclimits.bounded_semaphore_limit(semaphore)
    return None


@contextmanager
def async_features_compute_limit_context(limit: int) -> Generator[None, Any, None]:
    """Enforce a limit in a context."""
    token = async_features_compute_semaphore.set(BoundedSemaphore(limit))
    try:
        yield
    finally:
        async_features_compute_semaphore.reset(token)


@overload
async def amap_gather_with_limit[A, B](source: Iterable[A],
                                       mapper: Callable[[A], Coroutine[Any, Any, B]],
                                       return_exceptions: Literal[False] = False) -> list[B]: ...


@overload
async def amap_gather_with_limit[A, B](source: Iterable[A],
                                       mapper: Callable[[A], Coroutine[Any, Any, B]],
                                       return_exceptions: bool) -> list[B | BaseException]: ...


async def amap_gather_with_limit[A, B](source: Iterable[A],
                                       mapper: Callable[[A], Coroutine[Any, Any, B]],
                                       return_exceptions: bool = False) -> list:
    """As asyncio.gather, but run a limited amount of tasks at once, bound by the current context eval limit."""
    match async_features_compute_semaphore.get():
        case None:
            return await asyncio.gather(*(mapper(a) for a in source), return_exceptions=return_exceptions)
        case semaphore:
            return await asynclimits.amap_gather_bounded_concurrency(source, mapper,
                max_concurrent=semaphore, return_exceptions=return_exceptions)
