import asyncio
import logging
from typing import Literal, cast, overload
from collections.abc import Awaitable, Callable, Iterable
import queue

type AsyncRunnable[T] = Callable[[], Awaitable[T]]

_logger = logging.getLogger(__name__)

@overload
async def bounded_parallelism[T](runnables: Iterable[AsyncRunnable[T]], max_concurrent_tasks: int,
                                  return_exceptions: Literal[False] = False) -> list[T]: ...

@overload
async def bounded_parallelism[T](runnables: Iterable[AsyncRunnable[T]], max_concurrent_tasks: int,
                                  return_exceptions: bool = True) -> list[T | Exception]: ...

async def bounded_parallelism[T](runnables: Iterable[AsyncRunnable[T]], max_concurrent_tasks: int,
                                  return_exceptions: bool = False) -> list[T | Exception] | list[T]:
    """Run tasks in parallel, but limit the number of concurrent tasks to max_concurrent_tasks,
    and start tasks in the order of the input list.
    
    Args:
        runnables: List of async callables to run
        max_concurrent_tasks: Maximum number of concurrent tasks to run
        return_exceptions: If True, return exceptions as well as results in the output list; if False,
                           a task failing with an exception propagates to the caller,
                           and any tasks not yet started will not be started.
                           (See the parameter of the same name to asyncio.gather.)
        
    Returns:
        List of results from the runnables in the same order as input
    """
    if not runnables:
        return []
    
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    runnable_queue = queue.Queue[tuple[int, AsyncRunnable[T]]]()
    for index, runnable in enumerate(runnables):
        runnable_queue.put((index, runnable))
    results_queue = queue.PriorityQueue[tuple[int, T | Exception]]()
    abort = asyncio.Event()

    async def run_with_semaphore() -> None:
        while not runnable_queue.empty() and not abort.is_set():
            async with semaphore:
                try:
                    index, runnable = runnable_queue.get_nowait()
                except queue.Empty: # another task beat us to it
                    return
                try:
                    result = await runnable()
                    results_queue.put((index, result))
                except Exception as e:
                    if return_exceptions:   
                        results_queue.put((index, e))
                    else:
                        abort.set()
                        raise e
                    
    try:
        # Always pass return_exceptions=False here; run_with_semaphore will raise if it wants to abort early
        await asyncio.gather(*[run_with_semaphore() for _ in range(max_concurrent_tasks)])
    except Exception as e:
        abort.set()
        raise e

    results: list[T | Exception] = []
    while not results_queue.empty():
        index, result = results_queue.get()
        results.append(result)

    # Satisfy type checker
    if return_exceptions:
        return results
    else:
        return cast(list[T], results)
