import asyncio
from collections import defaultdict

import pytest

from agentune.analyze.util.asyncmap import HalfAsyncMap


@pytest.mark.asyncio
async def test_half_async_map_basic_functionality() -> None:
    """Test basic get_or_update, caching, and exception handling."""
    async_map = HalfAsyncMap[str, int]()
    call_count = 0
    
    async def producer() -> int:
        nonlocal call_count
        call_count += 1
        return 42
    
    # Basic functionality and caching
    result1 = await async_map.get_or_update('key1', producer)
    result2 = await async_map.get_or_update('key1', producer)  # Should use cache
    assert result1 == result2 == 42
    assert call_count == 1  # Producer called only once
    
    # Exception handling
    async def failing_producer() -> int:
        raise ValueError('failed')
    
    with pytest.raises(ValueError, match='failed'):
        await async_map.get_or_update('fail', failing_producer)


@pytest.mark.asyncio
async def test_half_async_map_concurrency() -> None:
    """Test concurrent same-key sharing and different-key separation."""
    async_map = HalfAsyncMap[str, int]()
    call_counts: defaultdict[str, int] = defaultdict(int)
    events: dict[str, asyncio.Event] = {k: asyncio.Event() for k in ['same', 'diff']}
    
    async def producer(k: str) -> int:
        call_counts[k] += 1
        await events[k].wait()
        return ord(k[0]) * 10
    
    # Start: 2 same-key, 1 different-key
    tasks = [
        asyncio.create_task(async_map.get_or_update('same', lambda: producer('same'))),
        asyncio.create_task(async_map.get_or_update('same', lambda: producer('same'))),
        asyncio.create_task(async_map.get_or_update('diff', lambda: producer('diff')))
    ]
    
    await asyncio.sleep(0.01)  # Let tasks start
    assert call_counts['same'] == 1  # Only 1 producer for 2 same-key calls
    assert call_counts['diff'] == 1  # 1 producer for different key
    
    # Complete and verify results
    events['same'].set()
    events['diff'].set()
    results = await asyncio.gather(*tasks)
    
    assert results[0] == results[1] == ord('s') * 10  # Same-key calls get same result
    assert results[2] == ord('d') * 10  # Different key gets different result


@pytest.mark.asyncio
async def test_half_async_map_advanced_features() -> None:
    """Test join_all_outstanding, context manager, and custom storage."""
    async_map = HalfAsyncMap[str, int]()
    
    # Test join_all_outstanding with mixed success/failure
    async def fail() -> int: raise ValueError('fail')
    async def work() -> int: return 42
    
    fail_task = asyncio.create_task(async_map.get_or_update('f', fail))
    work_task = asyncio.create_task(async_map.get_or_update('w', work))
    
    await async_map.join_all_outstanding()  # Should not raise
    
    with pytest.raises(ValueError, match='fail'):
        await fail_task
    assert await work_task == 42
    
    # Test context manager and custom storage
    custom_storage = {'existing': 999}
    async def producer1() -> int:
        await asyncio.sleep(0)
        return 0
    async def producer2() -> int:
        await asyncio.sleep(0)
        return 123
    
    async with HalfAsyncMap[str, int](storage=custom_storage) as custom_map:
        assert await custom_map.get_or_update('existing', producer1) == 999
        await custom_map.get_or_update('new', producer2)
        assert custom_storage['new'] == 123
