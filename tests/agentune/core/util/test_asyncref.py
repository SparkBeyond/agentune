import asyncio
import datetime
import time

import pytest

from agentune.api.base import RunContext
from agentune.core.util import asyncref


def test_store_asyncio_event_loop_no_context() -> None:
    assert asyncref.get_stored_asyncio_event_loop() is None
    with pytest.raises(RuntimeError):
        asyncref.current_asyncio_event_loop()

async def test_store_asyncio_event_loop() -> None:
    loop = asyncio.get_running_loop()

    assert asyncref.get_stored_asyncio_event_loop() is None
    assert asyncref.current_asyncio_event_loop() == (loop, True)

    assert (await asyncio.to_thread(asyncref.get_stored_asyncio_event_loop)) is None
    with pytest.raises(RuntimeError):
        await asyncio.to_thread(asyncref.current_asyncio_event_loop)

    async with asyncref.store_asyncio_event_loop():
        assert asyncref.get_stored_asyncio_event_loop() is loop
        assert asyncref.current_asyncio_event_loop() == (loop, True)

        assert (await asyncio.to_thread(asyncref.get_stored_asyncio_event_loop)) is loop
        assert (await asyncio.to_thread(asyncref.current_asyncio_event_loop)) == (loop, False)

    # Outside the context
    assert asyncref.get_stored_asyncio_event_loop() is None

    async with await RunContext.create():
        assert asyncref.get_stored_asyncio_event_loop() is loop, 'RunContext stores the loop'

    assert asyncref.get_stored_asyncio_event_loop() is None, 'RunContext un-stores the loop on exit'

async def test_on_timeout() -> None:
    event = asyncio.Event()
    with asyncref.on_timeout(datetime.timedelta(seconds=0.05), event.set):
        pass
    await asyncio.sleep(0.1)
    assert not event.is_set(), 'Timeout event was not set'

    with asyncref.on_timeout(datetime.timedelta(seconds=0.05), event.set):
        await asyncio.sleep(0.1)
    assert event.is_set(), 'Timeout event was set'

    def test_on_sync_thread() -> None:
        event = asyncio.Event()
        with asyncref.on_timeout(datetime.timedelta(seconds=0.05), event.set):
            pass
        time.sleep(0.1)
        assert not event.is_set(), 'Timeout event was not set'

        with asyncref.on_timeout(datetime.timedelta(seconds=0.05), event.set):
            time.sleep(0.1)
        assert event.is_set(), 'Timeout event was set'

    async with asyncref.store_asyncio_event_loop():
        await asyncio.to_thread(test_on_sync_thread)
