import contextlib
from collections.abc import Iterator
from typing import Any

from duckdb import DuckDBPyConnection


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

def results_iter(conn: DuckDBPyConnection, batch_size: int = 100) -> Iterator[tuple[Any, ...]]:
    # More efficient to call fetchmany() and then flatten
    while True:
        batch = conn.fetchmany(batch_size)
        if not batch:
            break
        yield from batch

