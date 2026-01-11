"""Provide sync code launched through to_thread with a context var giving access to the originating asyncio event loop."""
import asyncio
import contextlib
import contextvars
import datetime
import queue
import threading
from collections.abc import AsyncGenerator, Callable, Generator

_asyncio_event_loop = contextvars.ContextVar[asyncio.AbstractEventLoop | None]('asyncio_event_loop', default=None)

def get_stored_asyncio_event_loop() -> asyncio.AbstractEventLoop | None:
    """Return the currently stored asyncio event loop, or None."""
    return _asyncio_event_loop.get()

def current_asyncio_event_loop() -> tuple[asyncio.AbstractEventLoop, bool]:
    """Return the current asyncio event loop (if called on an asyncio thread), or the stored asyncio event loop.

    Raise a RuntimeError if none is stored and the current thread is not async.

    Async code should NOT call this function, it should call asyncio.get_running_loop(). However, we support
    the case where async code calls some sync code which in turn calls this function.

    Returns:
        The event loop, and a boolean which is True if we are on the event-loop thread or False otherwise.

    Raises:
        RuntimeError: if called on a non-asyncio thread and no event loop is stored.
                      The error type is chosen to be the same type that asyncio.get_running_loop() raises
                      if there is no current loop.
    """
    stored_loop = _asyncio_event_loop.get()
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if current_loop is not None:
        if current_loop is stored_loop or stored_loop is None:
            return current_loop, True
        else:
            raise RuntimeError('Called on async thread but a different event loop is stored')
    elif stored_loop is not None:
        return stored_loop, False
    else:
        raise RuntimeError('No event loop is stored and we are not on an async thread')

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

@contextlib.contextmanager
def on_timeout(timeout: datetime.timedelta, action: Callable[[], None]) -> Generator[None, None, None]:
    """Executes the callable on the async thread after a timeout, unless the context is exited first.

    DO NOT pass an async callable by mistake!

    The async loop is either the current one or the stored one.

    This function is thread-safe: it can be called from a worker thread (e.g. one created by asyncio.to_thread)
    and will correctly schedule the timeout on the event loop thread.
    """
    # If we're on the event loop thread, we can use call_later directly.
    # If we're on a different thread, we need to use call_soon_threadsafe to schedule the timer.
    loop, is_event_loop_thread = current_asyncio_event_loop()
    if is_event_loop_thread:
        timer = loop.call_later(timeout.total_seconds(), action)
        try:
            yield
        finally:
            timer.cancel()
    else:
        # We're on a worker thread. Use a thread-safe queue to pass the timer handle back.
        timer_queue: queue.SimpleQueue[asyncio.TimerHandle] = queue.SimpleQueue()
        timer_canceled = threading.Event()

        def schedule_timer() -> None:
            timer = loop.call_later(timeout.total_seconds(), action)
            timer_queue.put(timer)
        loop.call_soon_threadsafe(schedule_timer)

        def cancel_timer() -> None:
            timer = timer_queue.get()
            timer.cancel()
            timer_canceled.set()

        try:
            yield
        finally:
            # Cancel must also be done on the event loop thread
            loop.call_soon_threadsafe(cancel_timer)
            # Block until the timer is actually canceled; we guarantee the callback won't be called
            # after this context is exited
            timer_canceled.wait()
