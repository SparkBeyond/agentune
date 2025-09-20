import asyncio

import pytest

from agentune.analyze.util.asynclimits import (
    amap_gather_bounded_concurrency,
)
from agentune.analyze.util.atomic import AtomicInt


async def test_amap_gather_bounded_concurrency() -> None:
    """Test gather version returns complete list in order."""
    async def mapper(x: int) -> int:
        await asyncio.sleep(0.01)
        return x * 2
    
    source = list(range(1, 101))  # 100 items
    results = await amap_gather_bounded_concurrency(source, mapper, max_concurrent=10)
    
    # Results should maintain original order
    expected = [x * 2 for x in range(1, 101)]
    assert results == expected

async def test_concurrency_limit_respected() -> None:
    """Test that max_concurrent limit is actually respected and order preserved."""
    concurrent_count = AtomicInt(0)
    max_concurrent_seen = AtomicInt(0)
    
    async def mapper(x: int) -> int:
        current = concurrent_count.inc_and_get()
        max_concurrent_seen.setmax(current)
        await asyncio.sleep(0.01)  # Long enough to see concurrency
        concurrent_count.inc_and_get(-1)
        return x
    
    source = list(range(100))  # 100 items
    results = await amap_gather_bounded_concurrency(source, mapper, max_concurrent=5)
    assert max_concurrent_seen.get() == 5
    assert results == source  # Order preserved



async def test_error_handling_return_exceptions_false() -> None:
    """Test error propagation when return_exceptions=False."""
    async def mapper(x: int) -> int:
        if x == 50:  # Error in middle of 100 items
            raise ValueError(f'Error for {x}')
        await asyncio.sleep(0.01)
        return x * 2
    
    source = list(range(1, 101))  # 100 items
    
    with pytest.raises(ValueError, match='Error for 50'):
        await amap_gather_bounded_concurrency(source, mapper, max_concurrent=10, return_exceptions=False)



async def test_error_handling_return_exceptions_true() -> None:
    """Test error collection when return_exceptions=True, preserving order."""
    async def mapper(x: int) -> int:
        if x % 25 == 0:  # Errors at positions 25, 50, 75, 100
            raise ValueError(f'Error for {x}')
        await asyncio.sleep(0.01)
        return x * 2
    
    source = list(range(1, 101))  # 100 items
    results = await amap_gather_bounded_concurrency(source, mapper, max_concurrent=10, return_exceptions=True)
    assert len(results) == 100
    
    # Check that results are in correct order
    for i, result in enumerate(results, 1):
        if i % 25 == 0:
            assert isinstance(result, ValueError)
            assert f'Error for {i}' in str(result)
        else:
            assert result == i * 2



async def test_shared_semaphore() -> None:
    """Test that shared semaphore limits across multiple calls, preserving order."""
    concurrent_count = AtomicInt(0)
    max_concurrent_seen = AtomicInt(0)
    semaphore = asyncio.BoundedSemaphore(5)
    
    async def mapper(x: int) -> int:
        current = concurrent_count.inc_and_get()
        # Update max_concurrent_seen atomically
        while True:
            current_max = max_concurrent_seen.get()
            if current <= current_max:
                break
            max_concurrent_seen.put(current)
            break
        
        await asyncio.sleep(0.001)
        concurrent_count.inc_and_get(-1)
        return x
    
    # Start two concurrent amap_gather_bounded_concurrency calls with shared semaphore
    source1 = list(range(1, 51))   # 50 items
    source2 = list(range(51, 101)) # 50 items
    
    async def run_mapper(source: list[int]) -> list[int | Exception]:
        return await amap_gather_bounded_concurrency(source, mapper, max_concurrent=semaphore)
    
    results1, results2 = await asyncio.gather(run_mapper(source1), run_mapper(source2))
    
    assert max_concurrent_seen.get() == 5
    # Results should maintain order within each call
    assert results1 == source1
    assert results2 == source2


async def test_edge_cases() -> None:
    """Test edge cases: empty source, single item, and variable completion times."""
    async def mapper(x: int) -> int:
        # Make some items take longer to test ordering
        sleep_time = 0.1 if x % 10 == 0 else 0.0001
        await asyncio.sleep(sleep_time)
        return x * 2
    
    # Test empty source
    results = await amap_gather_bounded_concurrency([], mapper, max_concurrent=2)
    assert results == []
    
    # Test single item
    results = await amap_gather_bounded_concurrency([5], mapper, max_concurrent=2)
    assert results == [10]
    
    # Test variable completion times with order preservation
    source = list(range(1, 21))  # 20 items, every 10th takes longer
    results = await amap_gather_bounded_concurrency(source, mapper, max_concurrent=5)

    # Results should maintain original order despite different completion times
    expected = [x * 2 for x in range(1, 21)]
    assert results == expected
