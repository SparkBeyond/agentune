import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import Executor
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from duckdb import DuckDBPyConnection

from agentune.analyze.core.database import DuckdbManager
from agentune.analyze.llmcache.base import LLMCacheBackend
from agentune.analyze.llmcache.serializing_kvstore import SerializingKVStore
from agentune.analyze.llmcache.sqlite_lru import SqliteLru, threadlocal_connections
from agentune.analyze.run.base import RunContext, default_httpx_async_client
from agentune.analyze.util.lrucache import LRUCache


@pytest.fixture
def ddb_manager() -> Iterator[DuckdbManager]:
    """Provide a DuckdbManager connected to an in-memory database."""
    with contextlib.closing(DuckdbManager.in_memory()) as ddb_manager:
        yield ddb_manager


@pytest.fixture
def conn(ddb_manager: DuckdbManager) -> Iterator[DuckDBPyConnection]:
    """Provide an in-memory DuckDB connection."""
    with ddb_manager.cursor() as conn:
        yield conn


@pytest.fixture
async def run_context(ddb_manager: DuckdbManager, httpx_async_client: httpx.AsyncClient) -> AsyncIterator[RunContext]:
    """Create a default RunContext backed by an in-memory DuckDBManager."""
    async with contextlib.aclosing(RunContext.create_default_context(ddb_manager, httpx_async_client)) as run_context:
        yield run_context


@pytest.fixture
async def httpx_async_client() -> AsyncIterator[httpx.AsyncClient]:
    """Create an httpx client """
    async with default_httpx_async_client() as client:
        yield client

@pytest.fixture
async def executor() -> Executor:
    # Don't want to create another threadpool when async contexts will create one already.
    # Ugly hack to force the event loop to instantiate the threadpool executor.
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, lambda: None)
    return loop._default_executor # type: ignore[attr-defined]

@pytest.fixture
def sqlite_lru(tmp_path: Path) -> Iterator[SqliteLru]:
    file = tmp_path / 'cache.sqlite'
    with SqliteLru(file, 100_000, timedelta(seconds=1), threadlocal_connections()) as sqlite_lru:
        yield sqlite_lru

@pytest.fixture
def memory_llm_cache() -> LLMCacheBackend:
    return LRUCache(1000)

@pytest.fixture
def disk_llm_cache(sqlite_lru: SqliteLru) -> LLMCacheBackend:
    return SerializingKVStore(sqlite_lru)
