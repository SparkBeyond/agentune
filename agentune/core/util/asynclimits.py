import asyncio
from asyncio import BoundedSemaphore
from collections.abc import Callable, Coroutine, Iterable
from typing import Any, Literal, overload


def bounded_semaphore_limit(semaphore: BoundedSemaphore) -> int:
    return semaphore._bound_value # type: ignore [attr-defined]

@overload
def amap_gather_bounded_concurrency[A, B](source: Iterable[A],
                                          mapper: Callable[[A], Coroutine[Any, Any, B]],
                                          max_concurrent: int | BoundedSemaphore,
                                          return_exceptions: Literal[False] = False) -> Coroutine[Any, Any, list[B]]: ...

@overload
def amap_gather_bounded_concurrency[A, B](source: Iterable[A],
                                          mapper: Callable[[A], Coroutine[Any, Any, B]],
                                          max_concurrent: int | BoundedSemaphore, return_exceptions: bool) -> Coroutine[Any, Any, list[B | BaseException]]: ...


async def amap_gather_bounded_concurrency[A, B](source: Iterable[A],
                                                mapper: Callable[[A], Coroutine[Any, Any, B]],
                                                max_concurrent: int | BoundedSemaphore,
                                                return_exceptions: bool = False) -> list:
    """As asyncio.gather, but with a limited number of tasks active at once.

    Args:
        max_concurrent: passing the same semaphore instance to multiple invocations applies a shared limit to their concurrent tasks.
    """
    match max_concurrent:
        case BoundedSemaphore(): semaphore = max_concurrent
        case int(): semaphore = BoundedSemaphore(max_concurrent)

    result: list[tuple[int, B | BaseException]] = []

    try:
        async with asyncio.TaskGroup() as group:
            async def work(item: A, index: int) -> None:
                try:
                    b = await mapper(item)
                    result.append((index, b))
                except BaseException as e:
                    if return_exceptions:
                        result.append((index, e))
                    else:
                        raise
                finally:
                    semaphore.release()

            for index, item in enumerate(source):
                await semaphore.acquire()
                group.create_task(work(item, index))

    except BaseExceptionGroup as group:
        if len(group.exceptions) == 1:
            # Preserve behavior of asyncio.gather
            raise group.exceptions[0] from None
        else:
            raise

    return [result for _, result in sorted(result, key=lambda t: t[0])]

