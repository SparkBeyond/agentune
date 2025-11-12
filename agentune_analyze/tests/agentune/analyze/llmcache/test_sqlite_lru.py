import logging
import random
import time
from collections.abc import Iterator
from concurrent.futures import Executor
from datetime import timedelta
from pathlib import Path

import pytest

from agentune.analyze.llmcache.sqlite_lru import (
    ConnectionProviderFactory,
    SqliteLru,
    connection_pool,
    threadlocal_connections,
)

_logger = logging.getLogger(__name__)

@pytest.fixture(params=[threadlocal_connections(), connection_pool(10)])
def connection_provider_factory(request: pytest.FixtureRequest) -> ConnectionProviderFactory:
    return request.param

@pytest.fixture
def sqlite_lru(connection_provider_factory: ConnectionProviderFactory, tmp_path: Path) -> Iterator[SqliteLru]:
    file = tmp_path / 'cache.sqlite'
    with SqliteLru(file, 10000, timedelta(seconds=60), connection_provider_factory) as sqlite_lru:
        yield sqlite_lru

def test_sqlite_lru_sanity(sqlite_lru: SqliteLru) -> None:
    assert len(sqlite_lru) == 0
    assert b'key' not in sqlite_lru
    with pytest.raises(KeyError):
        _ = sqlite_lru[b'key']

    sqlite_lru[b'key'] = b'value'
    assert len(sqlite_lru) == 1
    assert sqlite_lru[b'key'] == b'value'
    assert b'key' in sqlite_lru
    assert b'key2' not in sqlite_lru

    sqlite_lru[b'key2'] = b'value2'
    assert len(sqlite_lru) == 2
    assert sqlite_lru[b'key2'] == b'value2'
    assert b'key2' in sqlite_lru

    del sqlite_lru[b'key']
    assert len(sqlite_lru) == 1
    assert b'key' not in sqlite_lru
    with pytest.raises(KeyError):
        _ = sqlite_lru[b'key']
    with pytest.raises(KeyError):
        del sqlite_lru[b'key']
    assert sqlite_lru[b'key2'] == b'value2'

    sqlite_lru[b'key2'] = b'new_value'
    assert len(sqlite_lru) == 1
    assert sqlite_lru[b'key2'] == b'new_value'

def lru_size(sqlite_lru: SqliteLru) -> int:
    with sqlite_lru.acquire() as conn:
        result = conn.execute('SELECT sum(length(value)) FROM cache').fetchone()
        match result:
            case (None,): return 0 # If table is empty
            case (int(size),): return size
            case _: raise ValueError(f'Unexpected query result {result}')

def test_sqlite_lru_eviction(tmp_path: Path, connection_provider_factory: ConnectionProviderFactory) -> None:
    file = tmp_path / 'cache.sqlite'
    with SqliteLru(file, 10_000, timedelta(0), connection_provider_factory) as sqlite_lru:
        # cleanup_interval=0 makes the cleanup thread run the cleanup query all the time

        keys = [str(i+1).encode() for i in range(200)] # Tested locally with higher numbers, but keeping this low to pass quickly on the CI
        values = [key.zfill(500) for key in keys]
        assert sum(len(value) for value in values) > sqlite_lru.maxsize

        keys_to_keep_hot = []

        expected_state: dict[bytes, bytes] = {}

        total_size = 0
        for key, value in zip(keys, values, strict=False):
            if key in [keys[1], keys[5], keys[30]]:
                keys_to_keep_hot.append(key)

            sqlite_lru[key] = value
            total_size += len(value)
            expected_state[key] = value

            # Touch hot keys to update atime
            for hot_key in keys_to_keep_hot:
                _ = sqlite_lru[hot_key] # Raises KeyError if key evicted; this asserts that they should not be evicted

            # Sleep 1ms so the clock advances, and different rows get different timestamps (with 1ms resolution)
            time1 = time.time()
            time.sleep(0.001)
            time2 = time.time()
            assert int(time2 * 1000) > int(time1 * 1000)

            sqlite_lru._cleanup_thread.wait_for_complete_run()

            if total_size <= sqlite_lru.maxsize:
                assert total_size == lru_size(sqlite_lru)
                assert expected_state == sqlite_lru._as_dict(), 'No items were evicted while below total size'
                #_logger.info(f'Inserted {key} with size {len(value)}, {total_size=}')
            else:
                #_logger.info(f'Over limit after inserting {key} with size {len(value)}')
                new_size = lru_size(sqlite_lru)
                assert new_size < total_size + len(value)
                assert new_size <= sqlite_lru.maxsize
                total_size = new_size
                expected_state = sqlite_lru._as_dict()

def test_sqlite_lru_multithreaded(tmp_path: Path, executor: Executor, connection_provider_factory: ConnectionProviderFactory) -> None:
    file = tmp_path / 'cache.sqlite'
    with SqliteLru(file, 10000, timedelta(seconds=1), connection_provider_factory) as sqlite_lru:
        # Values kept small to make this fast on the CI; I tested locally with higher values
        keys = tuple(str(i % 100).encode() for i in range(1000)) # keys % 100, so the same key sometimes sees different values
        values = tuple(str(i).encode().zfill(1000) for i in range(1000))
        items = tuple(zip(keys, values, strict=True))

        assert sum(len(b) for b in values) > sqlite_lru.maxsize

        def run_in_thread() -> None:
            rnd = random.Random(42)
            my_items = list(items)
            rnd.shuffle(my_items)
            start = time.time()
            for key, value in my_items:
                _ = sqlite_lru.get(key)
                sqlite_lru[key] = value
            elapsed = time.time() - start
            _logger.debug(f'{elapsed} seconds for {len(my_items)} items = {elapsed * 1000 / len(my_items) / 2} ms/op (50% mix read and write)')

        nthreads = 10
        futures = [executor.submit(run_in_thread) for _ in range(nthreads)]
        for future in futures:
            future.result() # Raises internal exception if any

        # Check that the cache is in a sane, consistent state after all this
        sqlite_lru._cleanup_thread.wait_for_complete_run()

        assert lru_size(sqlite_lru) <= sqlite_lru.maxsize

        expected_values: dict[bytes, list[bytes]] = {}
        for key, value in items:
            expected_values.setdefault(key, []).append(value)

        items_stored = sqlite_lru._as_dict()
        for key, value in items_stored.items():
            assert value in expected_values[key], f'For {key=}, expected one of {[len(x) for x in expected_values[key]]} but got {len(value)}'


