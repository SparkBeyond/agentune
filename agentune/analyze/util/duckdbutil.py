import contextlib
from collections.abc import AsyncIterator, Iterator
from typing import Any

from duckdb import DuckDBPyConnection, DuckDBPyRelation


@contextlib.contextmanager
def transaction_scope(conn: DuckDBPyConnection) -> Iterator[DuckDBPyConnection]:
    conn.begin()
    try:
        yield conn
        conn.commit()
    except:
        conn.rollback()
        raise


def read_results(conn: DuckDBPyConnection, batch_size: int = 100) -> list[tuple[Any, ...]]:
    result: list[tuple[Any, ...]] = []
    while True:
        more = conn.fetchmany(batch_size)
        if not more:
            return result
        result.extend(more)

def results_iter(src: DuckDBPyConnection | DuckDBPyRelation, batch_size: int = 100) -> Iterator[tuple[Any, ...]]:
    # More efficient to call fetchmany() and then flatten
    while True:
        batch = src.fetchmany(batch_size)
        if not batch:
            break
        yield from batch


@contextlib.asynccontextmanager
async def acursor(conn: DuckDBPyConnection) -> AsyncIterator[DuckDBPyConnection]:
    """Create a cursor which will be closed when the `async with` context manager exits."""
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        cursor.close()
