
import asyncio
import datetime
import time

import duckdb
import duckdb.sqltypes
import pytest
from duckdb import DuckDBPyConnection

from agentune.api.base import RunContext
from agentune.core.util import duckdbutil
from agentune.core.util.duckdbutil import ConnectionWithInit, RowidNature, conn_timeout


def test_connection_use(conn: DuckDBPyConnection) -> None:
    def current_schema(conn: DuckDBPyConnection | ConnectionWithInit) -> str:
        match conn.execute('select current_schema()').fetchone():
            case (schema,): return schema
            case other: raise ValueError(f'Unexpected result: {other}')
    try:

        assert current_schema(conn) == 'main'
        conn.execute('create schema myschema')
        conn.execute('use myschema')
        assert current_schema(conn) == 'myschema'

        cursor: DuckDBPyConnection | ConnectionWithInit
        with conn.cursor() as cursor:
            assert current_schema(cursor) == 'main', 'USE not preserved in cursors'

        wrapper = ConnectionWithInit.use(conn, 'myschema')
        assert current_schema(wrapper) == 'myschema'
        with wrapper.cursor() as cursor:
            assert isinstance(cursor, ConnectionWithInit)
            assert current_schema(cursor) == 'myschema'

            with cursor.cursor() as inner_cursor:
                assert current_schema(inner_cursor) == 'myschema'

            # Check that inner_cursor was closed when we exited the scope
            with pytest.raises(duckdb.ConnectionException, match='closed'):
                assert current_schema(inner_cursor)

            assert current_schema(cursor) == 'myschema', 'Previous connection is still open'
    finally:
        conn.execute('drop schema myschema cascade')

def test_rowid_nature(conn: DuckDBPyConnection) -> None:
    conn.execute('create table tab1(i int)')
    conn.execute('create table tab2(i int, rowid int)')

    assert duckdbutil.test_rowid_nature(conn, 'tab1') == RowidNature.PSEUDOCOLUMN
    assert duckdbutil.test_rowid_nature(conn, 'tab2') == RowidNature.COLUMN

    df = conn.table('tab1').pl()
    conn.register('df', df)
    assert duckdbutil.test_rowid_nature(conn, 'df') == RowidNature.NONE


async def test_with_timeout() -> None:
    async with await RunContext.create() as ctx:
        with ctx.db.cursor() as conn:
            def slow(i: int) -> int:
                time.sleep(0.01)
                return i + 1

            conn.create_function('slow', slow, [duckdb.sqltypes.INTEGER], duckdb.sqltypes.INTEGER)

            def timeit(timeout: datetime.timedelta) -> float:
                with conn_timeout(conn, timeout):
                    start = time.time()
                    conn.sql('select slow(t.i::integer) from unnest(range(10)) as t(i)').fetchall()
                    elapsed = time.time() - start
                    return elapsed

            # Have to do it on another thread so the current async thread isn't blocked from executing the timeout timer
            async def timeit_in_thread(timeout: datetime.timedelta) -> float:
                return await asyncio.to_thread(timeit, timeout)

            # Doesn't timeout with large enough timeout
            assert await timeit_in_thread(datetime.timedelta(seconds=1)) >= 0.1

            # Can't invoke directly on async thread
            with pytest.raises(RuntimeError, match='must not be called on an async thread'):
                with conn_timeout(conn, datetime.timedelta(seconds=0.01)):
                    pass

            # Works on sync thread
            with pytest.raises(duckdb.InterruptException):
                await timeit_in_thread(datetime.timedelta(seconds=0.01))



