# noqa: INP001
# The above disables the Ruff rule saying there should be an __init__.py in this directory.
# Adding an __init__.py here breaks pytest.

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import Executor
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from duckdb import DuckDBPyConnection

from agentune.api.base import RunContext
from agentune.api.defaults import create_default_httpx_async_client
from agentune.core.database import DuckdbManager
from agentune.core.llm import LLMContext
from agentune.core.llmcache.base import LLMCacheBackend
from agentune.core.llmcache.serializing_kvstore import SerializingKVStore
from agentune.core.llmcache.sqlite_lru import SqliteLru, threadlocal_connections
from agentune.core.openai import OpenAIProvider
from agentune.core.sercontext import SerializationContext
from agentune.core.util.lrucache import LRUCache


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
async def httpx_async_client() -> AsyncIterator[httpx.AsyncClient]:
    """Create an httpx client """
    async with create_default_httpx_async_client() as client:
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

@pytest.fixture
def test_data_dir() -> Path:
    return Path(__file__).parent / 'data'

@pytest.fixture
def test_data_conversations(test_data_dir: Path) -> dict[str, Path]:
    return {
        'main_csv': test_data_dir / 'conversations' / 'example_main.csv',
        'conversations_csv': test_data_dir / 'conversations' / 'example_conversations_secondary.csv'
    }

@pytest.fixture
def llm_context(httpx_async_client: httpx.AsyncClient, memory_llm_cache: LLMCacheBackend) -> LLMContext:
    return LLMContext(httpx_async_client, (OpenAIProvider(),), cache_backend=memory_llm_cache)

@pytest.fixture
def llm_context_nocache(httpx_async_client: httpx.AsyncClient) -> LLMContext:
    return LLMContext(httpx_async_client, (OpenAIProvider(),), cache_backend=None)

@pytest.fixture
def ser_context(llm_context: LLMContext) -> SerializationContext:
    return SerializationContext(llm_context)

@pytest.fixture
async def ctx() -> AsyncIterator[RunContext]:
    async with await RunContext.create() as ctx:
        yield ctx

