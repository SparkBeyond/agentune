import asyncio
import contextlib
import logging
from pathlib import Path
from typing import cast

import attrs
import duckdb
import pytest
from duckdb.duckdb import DuckDBPyConnection

from agentune.analyze.core.database import (
    ArtIndex,
    DuckdbConfig,
    DuckdbFilesystemDatabase,
    DuckdbInMemoryDatabase,
    DuckdbManager,
    DuckdbTable,
)

_logger = logging.getLogger(__name__)


def test_tables_indexes(conn: DuckDBPyConnection) -> None:
    conn.execute('CREATE TABLE tab (a INT, "quoted name" INT)')
    conn.execute('CREATE INDEX idx ON tab (a, "quoted name")')
    table = DuckdbTable.from_duckdb('tab', conn)
    assert table.indexes == (ArtIndex(name='idx', cols=('a', 'quoted name')),)

    new_index = attrs.evolve(cast(ArtIndex, table.indexes[0]), name='idx2')
    table2 = attrs.evolve(table, name='tab2', indexes=(new_index,))
    table2.create(conn)
    assert DuckdbTable.from_duckdb('tab2', conn) == table2

    table3 = attrs.evolve(table, schema=table.schema.drop('a'))
    table3.create(conn, if_not_exists=True)
    assert DuckdbTable.from_duckdb('tab', conn) == table # Did not replace
    with pytest.raises(duckdb.CatalogException, match='already exists'):
        table3.create(conn)
    with pytest.raises(duckdb.BinderException, match='does not have a column named "a"'):
        table3.create(conn, or_replace=True)

    table4 = attrs.evolve(table3, indexes=())
    table4.create(conn, or_replace=True)
    assert DuckdbTable.from_duckdb('tab', conn) == table4


def test_duckdb_manager(tmp_path: Path) -> None:
    dbpath = tmp_path / 'test.db'
    with duckdb.connect(dbpath) as conn:
        conn.execute('CREATE TABLE test (id INTEGER)')
        conn.execute('INSERT INTO test (id) VALUES (1)')
        
    with contextlib.closing(DuckdbManager.in_memory()) as ddb_manager:
        with ddb_manager.cursor() as conn:
            conn.execute('CREATE TABLE main (id INTEGER)')
            conn.execute('INSERT INTO main (id) VALUES (1)')

        ddb_manager.attach(DuckdbFilesystemDatabase(dbpath), name='testdb')
        
        def assert_correct() -> None:
            with ddb_manager.cursor() as conn:
                res = conn.sql('SELECT main.id id, testdb.test.id id2 FROM memory.main main JOIN testdb.test ON main.id = testdb.test.id')
                assert res.fetchall() == [(1, 1)]

        assert_correct()

        async def async_test() -> None:
            assert_correct()
            await asyncio.to_thread(assert_correct)

        asyncio.run(async_test())

        # Second in-memory database
        memory2 = DuckdbInMemoryDatabase()
        ddb_manager.attach(memory2, name='memory2')

        with ddb_manager.cursor() as conn:
            conn.execute('CREATE TABLE memory2.main (id INTEGER)')
            conn.execute('INSERT INTO memory2.main (id) VALUES (100)')

            res = conn.sql('SELECT id FROM main')
            assert res.fetchall() == [(1,)] # Goes to main database
            res = conn.sql('SELECT id FROM memory2.main')
            assert res.fetchall() == [(100,)] # Goes to memory2 database

        ddb_manager.detach('memory2')
        with ddb_manager.cursor() as conn:
            res = conn.sql('SELECT id FROM main')
            assert res.fetchall() == [(1,)] # Goes to main database
            
            with pytest.raises(duckdb.CatalogException, match='does not exist'):
                conn.sql('SELECT id FROM memory2.main')

def test_duckdb_manager_config() -> None:
    with duckdb.connect(':memory:') as conn:
        assert conn.sql("SELECT current_setting('python_enable_replacements')").fetchone() == (True, ), \
            "Sanity check of duckdb's own default"

    with contextlib.closing(DuckdbManager.in_memory()) as ddb_manager, ddb_manager.cursor() as conn:
        assert not DuckdbConfig().python_enable_replacements, \
            "Sanity check of what we're testing"
        assert conn.sql("SELECT current_setting('python_enable_replacements')").fetchone() == (False, ), \
            'Default value of setting set in DuckdbConnectionConfig overrides duckdb default'

        default_threads = cast(int, conn.sql("SELECT current_setting('threads')").fetchall()[0][0])
        assert default_threads > 1, "Sanity check of what we're testing (fails on a single core machine, sorry)"

    with (contextlib.closing(DuckdbManager.in_memory(DuckdbConfig(threads=1))) as ddb_manager,
          ddb_manager.cursor() as conn):
        assert conn.sql("SELECT current_setting('threads')").fetchone() == (1, ), \
            'Setting threads in DuckdbConfig works'

    with (contextlib.closing(DuckdbManager.in_memory(DuckdbConfig(kwargs={'threads': 1}))) as ddb_manager,
          ddb_manager.cursor() as conn):
        assert conn.sql("SELECT current_setting('threads')").fetchone() == (1, ), \
            'Setting threads in DuckdbConfig.kwargs works'
