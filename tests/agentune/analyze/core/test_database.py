import asyncio
import contextlib
import logging
import math
from pathlib import Path
from typing import cast

import attrs
import duckdb
import polars as pl
import pytest
from duckdb.duckdb import CatalogException, DuckDBPyConnection

from agentune.analyze.core import types
from agentune.analyze.core.database import (
    ArtIndex,
    DuckdbConfig,
    DuckdbFilesystemDatabase,
    DuckdbInMemoryDatabase,
    DuckdbManager,
    DuckdbName,
    DuckdbTable,
)
from agentune.analyze.core.dataset import Dataset, DatasetSink, DatasetSource
from agentune.analyze.core.schema import Field, Schema

_logger = logging.getLogger(__name__)


def test_tables_indexes(conn: DuckDBPyConnection) -> None:
    conn.execute('CREATE TABLE tab (a INT, "quoted name" INT)')
    conn.execute('CREATE INDEX idx ON tab (a, "quoted name")')
    table = DuckdbTable.from_duckdb('tab', conn)
    assert table.indexes == (ArtIndex(name=DuckdbName.qualify('idx', conn), cols=('a', 'quoted name')),)

    new_index = attrs.evolve(cast(ArtIndex, table.indexes[0]), name=DuckdbName.qualify('idx2', conn))
    table2 = attrs.evolve(table, name=DuckdbName.qualify('tab2', conn), indexes=(new_index,))
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

        df = pl.DataFrame({'id': [1]}) # noqa: F841
        with ddb_manager.cursor() as conn2:
            conn2.execute('SET python_enable_replacements = true;')
            conn2.execute('select * from df') # Works if enabled explicitly for a connection
        with ddb_manager.cursor() as conn3:
            with pytest.raises(CatalogException, match='Table with name df does not exist'):
                conn3.execute('select * from df') # Other connections are not affected by it being enabled on a previous connection

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

def test_qualified_names() -> None:
    assert str(DuckdbName('a b', 'c d', 'e.f')) == '"c d"."e.f"."a b"'

    def dotest(conn: DuckDBPyConnection, database_name: str, schema_name: str) -> None:
        table_name = DuckdbName('foo bar.baz', database_name, schema_name)
        conn.execute(f'CREATE TABLE {table_name} (id INTEGER)')
        with conn.cursor() as cursor:
            cursor.execute(f'USE "{database_name}"."{schema_name}"')
            table = DuckdbTable.from_duckdb('foo bar.baz', cursor)
            assert table.name == DuckdbName('foo bar.baz', database_name, schema_name)
            assert table.name == DuckdbName.qualify('foo bar.baz', cursor)

        table2 = DuckdbTable(
            DuckdbName('a.b', database_name, schema_name),
            Schema((Field('foo', types.int32),)),
            indexes=(ArtIndex(DuckdbName('my index', database_name, schema_name), ('foo',)),)
        )
        table2.create(conn)
        with conn.cursor() as cursor:
            cursor.execute(f'USE "{database_name}"."{schema_name}"')

            assert DuckdbTable.from_duckdb('a.b', cursor) == table2
            assert ArtIndex.from_duckdb('a.b', cursor) == (ArtIndex(DuckdbName.qualify('my index', cursor), ('foo',)), )

        dataset = Dataset.from_polars(pl.DataFrame({'id': [1, 2, 3]}))
        DatasetSink.into_duckdb_table(table_name).write(dataset.as_source(), conn)
        assert DatasetSource.from_table_name(table_name, conn).to_dataset(conn) == dataset

    # Test once on the current database, and once on a secondary one whose name needs to be
    # explicitly specified
    with contextlib.closing(DuckdbManager.in_memory()) as ddb_manager, ddb_manager.cursor() as conn:
        dotest(conn, 'memory', 'main')

    # Reconnect to get a clean main database
    with contextlib.closing(DuckdbManager.in_memory()) as ddb_manager, ddb_manager.cursor() as conn:
        ddb_manager.attach(DuckdbInMemoryDatabase(), name='memory two')
        conn.execute('CREATE SCHEMA "memory two"."custom schema"')
        dotest(conn, 'memory two', 'custom schema')


def test_casts(conn: DuckDBPyConnection) -> None:
    conn.execute('CREATE TABLE tab (a INT16, b FLOAT8, c VARCHAR)')
    conn.executemany('INSERT INTO tab VALUES (?, ?, ?)', [
        (1, 1.0, 'a'),
        (None, 2.0, 'b'),
        (3, math.nan, 'c')
    ])
    table = DuckdbTable.from_duckdb('tab', conn)
    dataset = DatasetSource.from_table(table).to_dataset(conn)

    table2 = table.alter_column_types({'a': types.float64, 'c': types.string}, conn)
    dataset2 =  DatasetSource.from_table(table2).to_dataset(conn)
    assert dataset2 != dataset
    assert dataset2 == dataset.cast({'a': types.float64})

    # Lossy default semantics of int to bool cast
    table3 = table2.alter_column_types({'a': types.boolean}, conn)
    dataset3 = DatasetSource.from_table(table3).to_dataset(conn)
    assert dataset3.data['a'].to_list() == [True, None, True]

    # This much even duckdb won't let us do
    with pytest.raises(duckdb.ConversionException):
        table3.alter_column_types({'c': types.EnumDtype('a', 'c')}, conn)
    assert DuckdbTable.from_duckdb('tab', conn) == table3, 'Did not do anything'

    # In lax mode, nulls will be substituted
    table4 = table3.alter_column_types({
        'a': types.int32,
        'c': types.EnumDtype('a', 'c')
    }, conn, set_invalid_to_null=True)
    dataset4 = DatasetSource.from_table(table4).to_dataset(conn)
    assert dataset4.data['a'].to_list() == [1, None, 1]
    assert dataset4.data['c'].to_list() == ['a', None, 'c']

def test_idempotent_close(ddb_manager: DuckdbManager) -> None:
    ddb_manager.close()
    ddb_manager.close()


def test_temp_schema(tmp_path: Path) -> None:
    db_file = tmp_path / 'db.db'
    with contextlib.closing(DuckdbManager.on_disk(db_file)) as ddb_manager, ddb_manager.cursor() as conn:
        temp_schema_name = DuckdbManager.temp_schema_name

        name1 = ddb_manager.temp_random_name('foo bar')
        name2 = ddb_manager.temp_random_name('foo bar')
        assert name1 != name2
        assert name1.schema == temp_schema_name
        assert name1.database == ddb_manager._main_database.default_name
        assert name1.name.startswith('foo bar')
        assert len(name1.name) > len('foo bar')

        conn.execute(f'create table {name1}(i int)')
        conn.execute(f'create table "{name1.name}"(i int)') # Same name in the main schema

        tables = conn.execute('select database_name, schema_name, table_name from duckdb_tables()').fetchall()
        assert set(tables) == {
            ('db', 'main', name1.name),
            ('db', temp_schema_name, name1.name),
        }

        ddb_manager.attach(DuckdbInMemoryDatabase(), name='newdb')
        schemas = conn.execute("select schema_name from duckdb_schemas() where database_name='newdb'").fetchall()
        assert schemas == [('main',)], 'Temp schema not created in secondary database'


    with duckdb.connect(db_file) as conn:
        schemas = conn.execute("select schema_name from duckdb_schemas() where database_name='db'").fetchall()
        assert schemas == [('main', )], f'{temp_schema_name} schema was deleted when closing the database'

        conn.execute(f'create schema {temp_schema_name}')
        conn.execute(f'create table db.{temp_schema_name}.tab(i int)')
        tables = conn.execute(f"select table_name from duckdb_tables() where schema_name='{temp_schema_name}'").fetchall()
        assert tables == [('tab',)]

    with contextlib.closing(DuckdbManager.on_disk(db_file)) as ddb_manager, ddb_manager.cursor() as conn:
        schemas = conn.execute("select schema_name from duckdb_schemas() where database_name='db'").fetchall()
        assert set(schemas) == { ('main', ), (temp_schema_name, ) }, f'{temp_schema_name} schema still exists'
        tables = conn.execute(f"select table_name from duckdb_tables() where schema_name='{temp_schema_name}'").fetchall()
        assert tables == [], 'Tables in the temp schema were dropped on startup'

def test_cheap_size(conn: DuckDBPyConnection) -> None:
    conn.execute('CREATE TABLE tab (a INT)')
    conn.execute('INSERT INTO tab(a) VALUES (1), (2), (3)')
    source = DuckdbTable.from_duckdb('tab', conn).as_source(1)
    assert source.cheap_size(conn) == 3
