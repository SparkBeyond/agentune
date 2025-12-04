import datetime
import logging
from collections.abc import Awaitable
from datetime import timedelta
from pathlib import Path

import attrs
import pytest
from duckdb import CatalogException, InvalidInputException

from agentune.api.base import LlmCacheInMemory, LlmCacheOnDisk, RunContext
from agentune.core.database import DuckdbInMemory, DuckdbManager, DuckdbOnDisk
from agentune.core.llm import LLMSpec
from agentune.core.llmcache.base import CachingLLMMixin, LLMCacheBackend
from agentune.core.util.lrucache import LRUCache

_logger = logging.getLogger(__name__)


async def test_runcontext_ddb(tmp_path: Path) -> None:
    async with await RunContext.create() as ctx:
        assert isinstance(ctx._ddb_manager._main_database, DuckdbInMemory)
        assert ctx.db.databases == {'memory': ctx._ddb_manager._main_database}
        with ctx.db.cursor() as conn:
            assert conn.execute('select current_database()').fetchone() == ('memory',)
            assert conn.execute('select path from duckdb_databases() where database_name = current_database()').fetchone() == (None,)

    on_disk = DuckdbOnDisk(tmp_path / 'duck.db')
    assert not on_disk.path.exists()

    async with await RunContext.create(on_disk) as ctx:
        assert ctx._ddb_manager._main_database is on_disk
        assert ctx.db.databases == {'duck': ctx._ddb_manager._main_database}
        assert on_disk.path.exists()
        with ctx.db.cursor() as conn:
            assert conn.execute('select current_database()').fetchone() == ('duck',)
            assert conn.execute('select path from duckdb_databases() where database_name = current_database()').fetchone() == (str(on_disk.path),)

            conn.execute('create table foo (i int)')
            conn.execute('insert into foo (i) values (1)')
            assert conn.execute('select count(*) from foo').fetchone() == (1,)

    assert on_disk.path.exists(), 'DB persists'
    async with await RunContext.create(attrs.evolve(on_disk, read_only=True)) as ctx:
        with ctx.db.cursor() as conn:
            assert conn.execute('select count(*) from foo').fetchone() == (1,), 'Table persists'
            with pytest.raises(InvalidInputException, match='attached in read-only mode'):
                conn.execute('insert into foo (i) values (1)')

    async with await RunContext.create(on_disk) as ctx:
        db2 = DuckdbOnDisk(tmp_path / 'duck2.db')
        ctx.db.attach(db2)
        assert ctx.db.databases == {'duck': on_disk, 'duck2': db2}
        with ctx.db.cursor() as conn:
            conn.execute('create table duck2.foo (i int)')
            conn.execute('insert into duck2.foo (i) values (1)')
        ctx.db.detach('duck2')
        with ctx.db.cursor() as conn:
            with pytest.raises(CatalogException, match='does not exist'):
                conn.execute('select * from duck2.foo')

async def test_runcontext_ddb_existing_manager(ddb_manager: DuckdbManager) -> None:
    async with await RunContext.create(ddb_manager) as _ctx:
        pass
    with ddb_manager.cursor() as conn:
        assert conn.execute('select current_database()').fetchone() == ('memory', ), 'Preexisting DuckdbManager was not closed'

@pytest.mark.integration
async def test_llmcache(tmp_path: Path) -> None:
    llm_spec = LLMSpec('openai', 'gpt-4.1-mini')
    llm_spec2 = LLMSpec('openai', 'gpt-4o')

    async def with_time[T](callable: Awaitable[T]) -> tuple[T, timedelta]:
        start = datetime.datetime.now()
        result = await callable
        end = datetime.datetime.now()
        return result, end - start

    prompt = 'The meaning of life is '

    async def test_with(cache_backend: LlmCacheInMemory | LlmCacheOnDisk | LLMCacheBackend) -> None:
        async with await RunContext.create(llm_cache=cache_backend) as ctx:
            assert ctx._llm_context.cache_backend is not None
            llm = ctx.llm.get(llm_spec)
            assert isinstance(llm, CachingLLMMixin)
            assert llm._cache is ctx._llm_context.cache_backend
            assert len(llm._cache) == 0

            response1, time1 = await with_time(llm.acomplete(prompt))
            assert time1 > datetime.timedelta(seconds=0.5)
            response2, time2 = await with_time(llm.acomplete(prompt))
            assert response2.text == response1.text
            assert time2 < datetime.timedelta(seconds=0.01)
            assert len(llm._cache) == 1

            llm2 = ctx.llm.get(llm_spec2)
            assert isinstance(llm2, CachingLLMMixin)
            assert llm2._cache is llm._cache

            response3, time3 = await with_time(llm2.acomplete(prompt))
            assert response3.text != response1.text
            assert time3 > datetime.timedelta(seconds=0.5)
            assert len(llm._cache) == 2

        async with await RunContext.create(llm_cache=cache_backend) as ctx:
            llm = ctx.llm.get(llm_spec)
            llm2 = ctx.llm.get(llm_spec2)
            assert isinstance(llm, CachingLLMMixin)
            assert isinstance(llm2, CachingLLMMixin)
            assert llm._cache is ctx._llm_context.cache_backend
            assert len(llm._cache) == 2

            response4, time4 = await with_time(llm.acomplete(prompt))
            assert response4.text == response1.text
            assert time4 < datetime.timedelta(seconds=0.01)

            response5, time5 = await with_time(llm2.acomplete(prompt))
            assert response5.text == response3.text
            assert time5 < datetime.timedelta(seconds=0.01)
            assert len(llm2._cache) == 2

    # Passing an LlmCacheInMemory (the default) creates a new cache per context
    await test_with(LRUCache(10))
    await test_with(LlmCacheOnDisk(tmp_path / 'cache.db', 10000))

async def test_disable_llmcache() -> None:
    async with await RunContext.create(llm_cache=None) as ctx:
        llm = ctx.llm.get(LLMSpec('openai', 'gpt-4.1-mini'))
        assert not isinstance(llm, CachingLLMMixin)
