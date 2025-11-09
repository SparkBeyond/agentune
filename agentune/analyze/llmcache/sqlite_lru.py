from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import threading
import time
from abc import abstractmethod
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from sqlite3 import Connection
from typing import Protocol, override
from weakref import WeakSet

from attrs import define, field, frozen

from agentune.analyze.llmcache.serializing_kvstore import SerializingKVStore
from agentune.analyze.util.asyncmap import KVStore
from agentune.analyze.util.pool import ThreadsafePool

_logger = logging.getLogger(__name__)

class ConnectionProvider(Protocol):
    """A way to manage and acquire connections."""

    @abstractmethod
    @contextlib.contextmanager
    def acquire(self) -> Iterator[Connection]:
        """Get a connection. Must be used as a context manager, 'returning' the connections by exiting the context.

        The connection will NOT commit or rollback on exit from the context manager; use `with connection` for that to happen.

        If an error is raised inside the context manager, the connection is closed (rolling back any transaction);
        that connection instance will not be returned again. This provides a way to deliberately discard a connection instance.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close all connections that are not currently in use (i.e. given out by `acquire`).

        Connections currently in use are closed when they are returned (i.e. on existing the acquire() context).
        No more connections will be created.
        """
        ...

type ConnectionOpener = Callable[[bool], Connection] # param: check_same_thread
type ConnectionProviderFactory = Callable[[ConnectionOpener], ConnectionProvider]

class ThreadLocalConnectionsClosedError(Exception):
    """Raised when the connection provider has been closed and no more connections may be acquired."""

@frozen
class _WeakReferenceableConnection:
    # sqlite3.Connection cannot itself be weak-referenced
    conn: Connection

@define
class ThreadLocalConnection(ConnectionProvider):
    """Creates a thread-local connection on demand."""
    opener: ConnectionOpener

    all_conns: WeakSet = field(init=False, factory=WeakSet)
    all_conns_lock: threading.Lock = field(init=False, factory=threading.Lock)
    closed: threading.Event = field(init=False, factory=threading.Event)
    _local: threading.local = field(init=False, factory=threading.local)

    def _close_and_discard(self, conn: _WeakReferenceableConnection) -> None:
        with contextlib.suppress(Exception):
            conn.conn.close()
        if hasattr(self._local, 'conn'):
            del self._local.conn
        with self.all_conns_lock:
            self.all_conns.discard(conn)

    @override
    @contextlib.contextmanager
    def acquire(self) -> Iterator[Connection]:
        if self.closed.is_set():
            raise ThreadLocalConnectionsClosedError
        try:
            conn = self._local.conn
        except AttributeError:
            conn = _WeakReferenceableConnection(self.opener(True))
            with self.all_conns_lock:
                self.all_conns.add(conn)
            self._local.conn = conn
        try:
            yield conn.conn
        except:
            self._close_and_discard(conn)
            raise
        else:
            if self.closed.is_set():
                self._close_and_discard(conn)

    @override
    def close(self) -> None:
        self.closed.set()
        with self.all_conns_lock:
            conns = list(self.all_conns)
        for conn in conns:
            if conn is not None: # weakref access
                self._close_and_discard(conn)


def connection_pool(size: int = os.cpu_count() or 4,
                    acquire_timeout: timedelta = timedelta(1)) -> ConnectionProviderFactory:
    """Keep a fixed-size pool of reusable connections."""
    def factory(opener: ConnectionOpener) -> ConnectionProvider:
        return ThreadsafePool(lambda: opener(False), size, acquire_timeout)
    return factory

def threadlocal_connections() -> ConnectionProviderFactory:
    """Keep a connection per thread (opened on first use from that thread)."""
    def factory(opener: ConnectionOpener) -> ConnectionProvider:
        return ThreadLocalConnection(opener)
    return factory

class _CleanupThread(threading.Thread):
    def __init__(self, atime_updates: dict[bytes, int], atime_updates_lock: threading.Lock, cleanup_interval: timedelta,
                 conn_provider: ConnectionProvider, maxsize: int) -> None:
        super().__init__(name='SqliteLRU writes', daemon=True)

        self.atime_updates = atime_updates
        self.atime_updates_lock = atime_updates_lock
        self.cleanup_interval = cleanup_interval
        self.conn_provider = conn_provider
        self.maxsize = maxsize

        self.shutdown = threading.Event()

        # These are for tests only
        self.run_lock = threading.Lock() # Acquired every time we run
        self.have_run = threading.Event() # Is set every time we finish running; lets tests wait until the next run

    def close(self) -> None:
        self.shutdown.set()

    def cleanup(self) -> None:
        with self.run_lock:
            with self.atime_updates_lock:
                atimes = self.atime_updates.copy()
                self.atime_updates.clear()
            if atimes:
                with self.conn_provider.acquire() as conn:
                    conn.executemany('update cache set atime=:atime where key=:key',
                                     tuple({'key': k, 'atime': v} for k, v in atimes.items()))
            with self.conn_provider.acquire() as conn:
                max_atime = conn.execute('''SELECT MAX(atime)
                                            FROM (SELECT atime,
                                                         SUM(length(value)) OVER (ORDER BY atime DESC) as cumsum
                                                  FROM cache)
                                            WHERE cumsum > ?''', (self.maxsize,)).fetchone()[0]
                # Without the commit, the delete statement is prone to fail with a 'database locked' sqlite3.OperationalError
                # (without waiting for the defined timeout trying to acquire the lock). Starting a new transaction
                # is for some reason better than upgrading the previous read lock to a write lock.
                conn.commit()
                conn.execute('delete from cache WHERE atime <= ?', (max_atime,))
                result = conn.execute('select changes()').fetchone()
                match result:
                    case None: deleted = 0
                    case (int(count),): deleted = count
                    case _: raise ValueError(f'Unexpected query result {result}')
            if atimes or deleted:
                _logger.debug(f'Completed cleanup run, updated {len(atimes)} atimes and deleted {deleted} rows targeting {self.maxsize=}')
            self.have_run.set()

    def wait_for_complete_run(self) -> None:
        """For tests only"""
        with self.run_lock:
            self.have_run.clear()
        self.have_run.wait()

    @override
    def run(self) -> None:
        while not self.shutdown.is_set():
            try:
                if self.shutdown.wait(self.cleanup_interval.total_seconds()):
                    break
                self.cleanup()
            except Exception:
                _logger.exception('Error in cache DB cleanup thread')
        # When shutting down, flush remaining atimes one last time
        self.cleanup()
        _logger.info(f'Cleanup thread exiting, spent a total of {time.thread_time()} seconds')


@frozen(eq=False, hash=False)
class SqliteLru(KVStore[bytes, bytes], ConnectionProvider):
    """An LRU max-size cache backed by an sqlite database.

    The maximum size is measured as the sum of lengths of the values in the cache; the lengths of the keys are ignored.
    (In practice, we only store hashes of the real LLMCacheKeys, so their sizes in the cache are tiny.)

    Starts an associated thread which periodically (every cleanup_interval) batch-writes all access time updates
    and deletes old rows to keep the database below the maximum size. New value inserts / upserts happen immediately
    when calling __setitem__, without going through the other thread.

    Args:
        path: the sqlite database file. Will be created and its schema initialized if it does not exist.
              In theory, several processes using this code can use the same file at once;
              in practice, this has not been tested and may result in application code failures due to competition
              for write locks.
        maxsize: the maximum size of the caches *values*, in bytes.
                 The database file will exceed this size somewhat.
        cleanup_interval: how often to flush access time updates and to delete enough records to make sure the database
                          is below the max size.
                          If the process dies unexpectedly, some access time updates may be lost, and the database may be
                          over the max size, but inserts of new cache entries will not be lost.
        connection_provider_factory: a factory function for a way to manage connections.
                                     Use one of the functions in this module to create this.
    """

    path: Path
    maxsize: int
    cleanup_interval: timedelta
    connection_provider_factory: ConnectionProviderFactory

    _atime_updates: dict[bytes, int] = field(init=False, factory=dict)
    _atime_updates_lock: threading.Lock = field(init=False, factory=threading.Lock)

    _connection_provider: ConnectionProvider = field(init=False)
    @_connection_provider.default
    def _connection_provider_init(self) -> ConnectionProvider:
        return self.connection_provider_factory(self._open_connection)

    _cleanup_thread: _CleanupThread = field(init=False)
    @_cleanup_thread.default
    def _cleanup_thread_init(self) -> _CleanupThread:
        return _CleanupThread(self._atime_updates, self._atime_updates_lock, self.cleanup_interval, self, self.maxsize)

    def __attrs_post_init__(self) -> None:
        with self.acquire() as conn:
            conn.execute('''create table if not exists 
                            cache(key blob primary key not null, value blob not null, atime int not null) strict;''')
        self._cleanup_thread.start()

    def _open_connection(self, check_same_thread: bool) -> Connection:
        conn = sqlite3.connect(self.path, autocommit=False, check_same_thread=check_same_thread, isolation_level='EXCLUSIVE')
        # Can't do this inside a transaction, and normally there's always a transaction, so we have to end it
        # and then start a new one after we're done
        conn.execute('commit')
        conn.execute('PRAGMA journal_mode=WAL') # Technically only needed when first creating the database
        conn.execute('PRAGMA synchronous=normal') # Default is 'full' and that is slow and unnecessary in WAL mode
        conn.execute('begin')
        return conn

    @override
    @contextlib.contextmanager
    def acquire(self) -> Iterator[Connection]:
        with self._connection_provider.acquire() as _conn, _conn as conn:
            # Using the connection itself as a context makes the connection commit or rollback on exiting the context
            yield conn

    @override
    def __len__(self) -> int:
        with self.acquire() as conn:
            return conn.execute('select count(*) from cache').fetchone()[0]

    @override
    def __getitem__(self, key: bytes) -> bytes:
        with self.acquire() as conn:
            match conn.execute('select value from cache where key=?', (key,)).fetchone():
                case None: raise KeyError
                case (bytes(value),):
                    with self._atime_updates_lock:
                        self._atime_updates[key] = int(time.time() * 1000)
                    return value
                case other: raise ValueError(f'Unexpected query result {other}')

    @override
    def __setitem__(self, key: bytes, value: bytes) -> None:
        with self.acquire() as conn:
            conn.execute('insert or replace into cache (key, value, atime) values (?, ?, ?)',
                         (key, value, int(time.time() * 1000)))

    @override
    def __delitem__(self, key: bytes) -> None:
        with self.acquire() as conn:
            # In sqlite, delete/insert/update statements don't return the count, they always return nothing
            # and we have to fetch the count separately
            conn.execute('delete from cache where key=?', (key,))
            match conn.execute('select changes()').fetchone():
                case (0, ): raise KeyError
                case (1, ): pass
                case other: raise ValueError(f'Unexpected query result {other}')

    def close(self) -> None:
        self._cleanup_thread.close()
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join()
        self._connection_provider.close()

    def __enter__(self) -> SqliteLru:
        return self

    def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        self.close()

    def _as_dict(self) -> dict[bytes, bytes]:
        """Return the entire DB contents."""
        with self.acquire() as conn:
            results = conn.execute('select key, value from cache').fetchall()
            return dict(results)

    @contextlib.contextmanager
    @staticmethod
    def open_wrapped(path: Path, maxsize: int, cleanup_interval: timedelta = timedelta(seconds=60),
                     connection_provider_factory: ConnectionProviderFactory = threadlocal_connections()) -> Iterator[SerializingKVStore]:
        """Open or create an sqlite database to use as a cache, and wrap it to de/serialize LLM cache values.

        An SqliteLru instance MUST be closed, to stop the associated thread and close all connections.
        Use this function's return value as a context manager to do so.
        """
        sqlite_lru = SqliteLru(path, maxsize, cleanup_interval, connection_provider_factory)
        try:
            yield SerializingKVStore(sqlite_lru)
        finally:
            sqlite_lru.close()
