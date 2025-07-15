import asyncio
import contextlib
import logging
from pathlib import Path

import duckdb
import pytest

from agentune.analyze.flow.duckdb import DuckdbFilesystemDatabase, DuckdbManager

_logger = logging.getLogger(__name__)


def test_duckdb_manager(tmp_path: Path) -> None:
    dbpath = tmp_path / 'test.db'
    with contextlib.closing(duckdb.connect(dbpath)) as conn:
        conn.sql('CREATE TABLE test (id INTEGER)')
        conn.sql('INSERT INTO test (id) VALUES (1)')
        
    with contextlib.closing(DuckdbManager.create('test-threading')) as ddb_manager:
        with contextlib.closing(ddb_manager.cursor()) as conn:
            conn.sql('CREATE TABLE main (id INTEGER)')
            conn.sql('INSERT INTO main (id) VALUES (1)')

        ddb_manager.attach(DuckdbFilesystemDatabase('testdb', dbpath))
        
        def assert_correct(conn: duckdb.DuckDBPyConnection) -> None:
            res = conn.sql('SELECT main.id id, testdb.test.id id2 FROM main JOIN testdb.test ON main.id = testdb.test.id')
            assert res.fetchall() == [(1, 1)]

        assert_correct(ddb_manager.cursor())

        async def async_test() -> None:
            assert_correct(ddb_manager.cursor())
            await asyncio.to_thread(assert_correct, ddb_manager.cursor())

        asyncio.run(async_test())

        with contextlib.closing(DuckdbManager(ddb_manager.name)) as ddb_manager_copy:
            assert_correct(ddb_manager_copy.cursor()) # Previous databases still attached
            
            # Detaching affects all connections to that memory database
            ddb_manager.detach('testdb')
            with pytest.raises(duckdb.CatalogException):
                assert_correct(ddb_manager_copy.cursor()) 
            with pytest.raises(duckdb.CatalogException):
                assert_correct(ddb_manager.cursor()) 

            ddb_manager_copy.attach(DuckdbFilesystemDatabase('testdb', dbpath))
            assert_correct(ddb_manager_copy.cursor()) # Reattached
            assert_correct(ddb_manager.cursor()) 
