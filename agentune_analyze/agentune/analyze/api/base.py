"""High-level fluent API for using the library."""
from __future__ import annotations

import contextlib
import typing
from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import Path

import httpx
from attrs import frozen
from llama_index.core.base.llms.types import ChatResponse, CompletionResponse

from agentune.analyze.core.database import (
    DuckdbConfig,
    DuckdbDatabase,
    DuckdbInMemory,
    DuckdbManager,
)
from agentune.analyze.core.llm import LLMContext
from agentune.analyze.core.sercontext import SerializationContext
from agentune.analyze.llmcache import sqlite_lru
from agentune.analyze.llmcache.base import LLMCacheBackend, LLMCacheKey
from agentune.analyze.llmcache.sqlite_lru import ConnectionProviderFactory, SqliteLru
from agentune.analyze.util.lrucache import LRUCache

if typing.TYPE_CHECKING:
    from .data import BoundData
    from .db import BoundDb
    from .defaults import BoundDefaults
    from .json import BoundJson
    from .llm import BoundLlm
    from .ops import BoundOps

# These classes need to be in the base module because we use them to define the default values of parameters to create()
# and so we can't import them only when we need them

@frozen
class LlmCacheInMemory:
    """Create a new in-memory cache for LLM responses, holding up to `maxsize` items (i.e. request-response pairs)
    regardless of their size.
    """
    maxsize: int


@frozen
class LlmCacheOnDisk:
    """Open or create an on-disk cache file for LLM responses, holding up to `maxbytes` bytes of data.

    The cache is in SQLite format and uses a single file.

    Using the same cache file from multiple processes at once is possible but not recommended.
    """
    path: Path | str
    maxbytes: int
    cleanup_interval: timedelta = timedelta(seconds=60)
    connection_provider_factory: ConnectionProviderFactory = sqlite_lru.threadlocal_connections()


@frozen
class RunContext:
    """The entrypoint to the agentune-analyze API; sets up and manages resources and provides a fluent API to access them.

    Instances should be created using the `create()` static method. They can be used as a context manager
    (`async with await RunContext.create() as ctx:`).

    Exiting the context tears down and frees the resources, including closing on-disk databases, freeing memory,
    stopping threads and closing persistent HTTP connections. You should always make sure to close context instances
    (or terminate the Python process) and avoid creating multiple context instances at the same time.

    This class's API is organized into several namespaces, so that typical method calls look like `ctx.data.csv(...)`
    or `ctx.ops.analyze(...)`. The intermediate values (ctx.data and ctx.ops) don't do anything besides group methods.

    Methods whose name is a noun don't have side effects and require at most a small constant time to execute.
    Methods whose name is a verb can take arbitrarily long and have side effects.
    For example, `ctx.data.csv()` defines how to read a CSV file but doesn't actually read it,
    while `ctx.data.csv().copy_to_table()` reads it and copies it to a table, 'copy' being a verb.

    Most of the methods in this API need to access some of the resources managed by the context instance.
    To enable this, some methods return dataclasses called BoundXxx (e.g. BoundTable, BoundDataSource), which combine
    an existing entity type (the Xxx) with a run_context instance. The methods of a BoundXxx often parallel the methods
    of the underlying Xxx, omitting the parameters provided by the context (e.g. the duckdb connection).
    """

    _ser_context: SerializationContext
    _ddb_manager: DuckdbManager
    _components: AsyncExitStack

    @property
    def _llm_context(self) -> LLMContext:
        return self._ser_context.llm_context

    @property
    def _httpx_async_client(self) -> httpx.AsyncClient:
        return self._llm_context.httpx_async_client

    async def __aenter__(self) -> RunContext:
        return self

    async def __aexit__(self, _exc_type: type[BaseException] | None, _exc_value: BaseException | None, _traceback: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Free all resources and services managed by this instance, close files, etc. All methods will stop working."""
        await self._components.aclose()

    @staticmethod
    async def create(duckdb: DuckdbDatabase | DuckdbManager = DuckdbInMemory(),
                     duckdb_config: DuckdbConfig = DuckdbConfig(),
                     httpx_async_client: httpx.AsyncClient | None = None,
                     llm_cache: LlmCacheInMemory | LlmCacheOnDisk | LLMCacheBackend | None = LlmCacheInMemory(1000),
                     ) -> RunContext:
        """Create a new context instance (see the class doc). Remember to close it when you are done, by using it as
        a context manager or by calling the aclose() method explicitly.

        When existing component instances are passed in (duckdb, httpx_async_client, llm_cache), they are not closed
        when this context is closed. When we create new components based on other parameter values
        (e.g. llm_cache=LlmCacheOnDisk), they are closed when this context is closed.

        Args:
            duckdb: specify a primary database to connect to, either in-memory (DuckdbInMemory) or on-disk (DuckdbOnDisk).
                    When the context is closed, in-memory databases will be gone irretrievably.
                    You can attach and detach more databases later by calling the `db.attach` method,
                    but the primary database cannot be changed or detached.
                    If a custom DuckdbManager instance is passed, it will not be closed when this context is closed.
            duckdb_config: global settings for duckdb, such as memory limits and number of threads.
                           Ignored if a custom DuckdbManager instance is passed in `duckdb`.
            httpx_async_client: will be used for all HTTP requests, which are primarily LLM calls.
                                Call `agentune.analyze.api.defaults.create_default_httpx_async_client` to configure
                                a client's resource limits. If not specified, a client will be created
                                with the default arguments to create_default_httpx_async_client(), and it will be
                                torn down when the context is closed.
                                If a custom client instance is passed, it will not be closed when this context is closed.
            llm_cache: enables caching and reusing responses to LLM calls.
                       Can be stored in memory (LlmCacheInMemory) or on disk (LlmCacheOnDisk). An in-memory cached
                       will be discarded when the context is closed. An on-disk cache holds a file open until the
                       context is closed.
        """
        components = AsyncExitStack()

        match duckdb:
            case DuckdbDatabase() as db:
                ddb_manager = components.enter_context(contextlib.closing(DuckdbManager(db, duckdb_config)))
            case DuckdbManager():
                ddb_manager = duckdb

        if not httpx_async_client:
            from .defaults import create_default_httpx_async_client
            httpx_async_client = create_default_httpx_async_client()
            await components.enter_async_context(httpx_async_client)

        match llm_cache:
            case LlmCacheInMemory(maxsize):
                llm_cache_backend: LLMCacheBackend | None = LRUCache[LLMCacheKey, CompletionResponse | ChatResponse](maxsize)
            case LlmCacheOnDisk(path2, maxbytes, cleanup_interval, connection_provider_factory):
                if isinstance(path2, str):
                    path2 = Path(path2)
                llm_cache_backend = components.enter_context(SqliteLru.open_wrapped(path2, maxbytes, cleanup_interval,
                                                                                    connection_provider_factory))
            case _:
                llm_cache_backend = llm_cache

        ser_context = SerializationContext(LLMContext(httpx_async_client, cache_backend=llm_cache_backend))

        return RunContext(ser_context, ddb_manager, components)

    @property
    def data(self) -> BoundData:
        """Methods for reading and writing data."""
        from .data import BoundData
        return BoundData(self)

    @property
    def defaults(self) -> BoundDefaults:
        """Methods for accessing default settings and creating component instances with default settings."""
        from .defaults import BoundDefaults
        return BoundDefaults(self)

    @property
    def db(self) -> BoundDb:
        """Methods for accessing and manipulating the duckdb database."""
        from .db import BoundDb
        return BoundDb(self)

    @property
    def json(self) -> BoundJson:
        """Methods for reading and writing agentune values as JSON."""
        from .json import BoundJson
        return BoundJson(self)

    @property
    def ops(self) -> BoundOps:
        """Methods for running long-lived operations such as analyze and enrich."""
        from .ops import BoundOps

        return BoundOps(self)

    @property
    def llm(self) -> BoundLlm:
        """Methods for provisioning and creating LLM instances."""
        from .llm import BoundLlm
        return BoundLlm(self)



