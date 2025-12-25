import duckdb
import pytest
from duckdb import DuckDBPyConnection

from agentune.core.util import duckdbutil
from agentune.core.util.duckdbutil import ConnectionWithInit, RowidNature


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

