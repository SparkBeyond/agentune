import asyncio
from collections.abc import Callable, Coroutine, Sequence
from typing import Any


class OneAtATime:
    """Runs one sync task at a time via asyncio.to_thread().
    
    When asked to run another task, if the previous task is still running, it will wait for that one to finish.
    """

    def __init__(self) -> None:
        self.task: Coroutine[Any, Any, Any] | None = None

    async def start_running[T](self, func: Callable[..., T], *args: Any) -> Coroutine[T, Any, Any]:
        if self.task is not None:
            await self.task
            self.task = None
        self.task = asyncio.to_thread(func, *args)
        return self.task

    async def await_current(self) -> Any:
        if self.task is not None:
            await self.task
            self.task = None
    
async def one_at_a_time(funcs: Sequence[Callable]) -> None:
    """Runs one sync callable at a time via asyncio.to_thread()."""
    one_sync_task = OneAtATime()
    for func in funcs:
        _ = await one_sync_task.start_running(func)
    await one_sync_task.await_current()
