import asyncio
import contextlib
import logging
from pathlib import Path
from typing import cast

import attrs
import duckdb
import pytest

from agentune.analyze.core.database import (
    ArtIndex,
    DuckdbFilesystemDatabase,
    DuckdbInMemoryDatabase,
    DuckdbManager,
    DuckdbTable,
)

_logger = logging.getLogger(__name__)


def test_tables_indexes() -> None:
    with contextlib.closing(DuckdbManager.in_memory()) as ddb_manager, ddb_manager.cursor() as conn:
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
    with contextlib.closing(duckdb.connect(dbpath)) as conn:
        conn.sql('CREATE TABLE test (id INTEGER)')
        conn.sql('INSERT INTO test (id) VALUES (1)')
        
    with contextlib.closing(DuckdbManager.in_memory()) as ddb_manager:
        with contextlib.closing(ddb_manager.cursor()) as conn:
            conn.sql('CREATE TABLE main (id INTEGER)')
            conn.sql('INSERT INTO main (id) VALUES (1)')

        ddb_manager.attach(DuckdbFilesystemDatabase(dbpath), name='testdb')
        
        def assert_correct(conn: duckdb.DuckDBPyConnection) -> None:
            res = conn.sql('SELECT main.id id, testdb.test.id id2 FROM memory.main main JOIN testdb.test ON main.id = testdb.test.id')
            assert res.fetchall() == [(1, 1)]

        assert_correct(ddb_manager.cursor())

        async def async_test() -> None:
            assert_correct(ddb_manager.cursor())
            await asyncio.to_thread(assert_correct, ddb_manager.cursor())

        asyncio.run(async_test())

        # Second in-memory database
        memory2 = DuckdbInMemoryDatabase()
        ddb_manager.attach(memory2, name='memory2')

        with contextlib.closing(ddb_manager.cursor()) as conn:
            conn.execute('CREATE TABLE memory2.main (id INTEGER)')
            conn.execute('INSERT INTO memory2.main (id) VALUES (100)')

            res = conn.sql('SELECT id FROM main')
            assert res.fetchall() == [(1,)] # Goes to main database
            res = conn.sql('SELECT id FROM memory2.main')
            assert res.fetchall() == [(100,)] # Goes to memory2 database

        ddb_manager.detach('memory2')
        with contextlib.closing(ddb_manager.cursor()) as conn:
            res = conn.sql('SELECT id FROM main')
            assert res.fetchall() == [(1,)] # Goes to main database
            
            with pytest.raises(duckdb.CatalogException, match='does not exist'):
                conn.sql('SELECT id FROM memory2.main')

       
