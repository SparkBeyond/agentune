import asyncio
import contextlib
from collections.abc import Iterator, Callable, AsyncIterator, Coroutine
from typing import Any, Never

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

async def interrupt_in(conn: DuckDBPyConnection, timeout: float) -> Coroutine:
    '''Interrupt this connection after the given timeout, unless the context is exited first.

    Interrupting a connection causes an error to be raised in any threads that are blocked on reading the next row.

    If the result set is already computed and cached, it does NOT prevent calls to conn.fetch* from finishing reading it,
    so if your python code reads the results with slow computation in between the rows, this timeout isn't going to
    abort that. It only applies to queries where a fetch* call blocks.
    '''
    await asyncio.sleep(timeout)
    conn.interrupt()

@contextlib.asynccontextmanager
async def in_scope(coroutine: Coroutine) -> AsyncIterator[Never]:
    task = asyncio.create_task(cancel())
    try:
        yield
    finally:
        task.cancel()

@contextlib.contextmanager
def in_sync_scope(loop: asyncio.AbstractEventLoop, coroutine: Coroutine) -> Iterator[Never]:
    task = loop.create_task(coroutine)
    try:
        yield
    finally:
        task.cancel()

# How to use this:

async def example1(conn: DuckDBPyConnection) -> None:
    """Directly dispatch sync query from async context"""
    def sync_proces():
        return conn.execute("SELECT 1").fetchall()

    async with in_scope(interrupt_in(conn, 10)):
        return await asyncio.to_thread(sync_proces)

async def example2(conn: DuckDBPyConnection) -> None:
    """Work with interface which hides the query. The interface might create a cursor internally.
    We can't automatically interrupt all cursors created from a parent connection.
    """
    def sync_proces(loop: asyncio.AbstractEventLoop):
        with conn.cursor() as cursor:
            with in_sync_scope(loop, interrupt_in(cursor, 10)):
                return cursor.execute("SELECT 1").fetchall()

    return await asyncio.to_thread(sync_proces, asyncio.get_running_loop())

