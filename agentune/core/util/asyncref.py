"""Provide sync code launched through to_thread with a context var giving access to the originating asyncio event loop."""
import asyncio
import contextlib
import contextvars
from collections.abc import AsyncGenerator

_asyncio_event_loop = contextvars.ContextVar[asyncio.AbstractEventLoop | None]('asyncio_event_loop', default=None)

def get_current_asyncio_event_loop() -> asyncio.AbstractEventLoop | None:
    return _asyncio_event_loop.get()

def _current_asyncio_event_loop() -> asyncio.AbstractEventLoop:
    loop = _asyncio_event_loop.get()
    if loop is None:
        raise ValueError('No asyncio event loop stored. Consider using `with store_asyncio_event_loop()`')
    return loop

@contextlib.asynccontextmanager
async def store_asyncio_event_loop() -> AsyncGenerator[None, None]:
    """Store the current asyncio event loop in a context var that makes it available to sync code dispatched with to_thread.

    This function is async only to underline that it must run in an async context.
    """
    loop = asyncio.get_running_loop()
    current = _asyncio_event_loop.get()
    if current is not None and current is not loop:
        raise ValueError('A different event loop is already stored. There should only ever be one event loop.')
    _asyncio_event_loop.set(loop)
    try:
        yield
    finally:
        _asyncio_event_loop.set(None)

