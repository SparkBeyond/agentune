import contextlib
from collections.abc import Iterator
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


def results_iter(src: DuckDBPyConnection | DuckDBPyRelation, batch_size: int = 100) -> Iterator[tuple[Any, ...]]:
    # More efficient to call fetchmany() and then flatten
    while True:
        batch = src.fetchmany(batch_size)
        if not batch:
            break
        yield from batch


