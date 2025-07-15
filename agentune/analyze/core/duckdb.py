from __future__ import annotations

import threading
from pathlib import Path

import duckdb
from attrs import frozen

import agentune.analyze.core.setup
from agentune.analyze.util.atomic import AtomicInt


@frozen
class DuckdbFilesystemDatabase:
    """A file-backed database we can create or open. Does not represent an open connection."""

    name: str # Name to attach as
    path: Path # Filesystem path to open
    read_only: bool = True

class DuckdbManager:
    """Manages duckdb databases and connections, and relatedly, the size and creation of the duckbd threadpool(s).

    -----------------------

    Duckdb behaves like this:
    1. Every call to duckdb.connect(target) for a new target creates a new 'database instance', which includes some 
       per-DB-instance settings and state, and a threadpool (whose size can then be changed).
       
       Calling duckdb.connect(target) for a target that already has a live connection in the process returns 
       a new connection to the same database instance; this works until all connections to the database instance
       are explicitly closed or are garbage collected.

    2. Once you have a connection, you can call its .cursor() method to get another connection to the same DB instance.
       This is cheap.

       Connections are blocking, so at minimum we need a cursor per thread. Also, some things happen at connection 
       scope, and it's useful to create connections to scope various effects, like USE statements.

    3. An on-disk database, but NOT an in-memory database, can also be attached to an existing connection.
       (Note that the scope of ATTACH is per connection, not per database instance!)
       
       Therefore, a connection can have at most one in-memory database attached (if the original connect() call was
       to an in-memory database), and zero or more on-disk ones.

    4. Each database (=catalog) can have multiple schemas, as normal in SQL. 
       See https://duckdb.org/docs/stable/sql/statements/attach#name-qualification

       This becomes relevant when attaching many databases that potentially have the same 'default' name.

    5. Attaching/detaching databases affects existing connections created via cursor() calls from each other;
       only connections created by calling duckdb.connect() are unaffected.

    ----------------

    Each instance of this class provides access to an in-memory database, and zero or more on-disk databases.
    The default database is the in-memory one (you can change this by running USE on a connection).

    The class instance always keeps a connection instance, so closing all connections outside of this class
    will not discard the in-memory database or free resources until this class's own .close() is called.
    Code SHOULD do its best to scope use of connection instances obtained from this class so that they ARE
    closed fairly deterministically.
    
    Calling the .attach and .detach methods affects all connection instances previously returned by cursor(), 
    and indeed all connections returned by duckdb.connect() to the same memory database.
    It is therefore important not to reuse attached database names, or else to use different schemas.
    
    There is currently no way to force close a database instance (discard an in-memory database, release files, close threads) 
    while live (python) connections remain. We can implement this in the future by using lower-level duckdb APIs.
    
    (TODO: allow reusing the in-memory database between class instances, since a separate threadpool is created for each one, 
    and separate the class instances by using different schemas.)
    """

    def __init__(self, name: str):
        agentune.analyze.core.setup.setup()

        self.name = name
        self._conn = duckdb.connect(f':memory:{name}')
        self._conn.load_extension('spatial')
        self._databases: dict[str, DuckdbFilesystemDatabase] = {}
        self._lock = threading.Lock()

    _instance_counter = AtomicInt()
    
    @staticmethod
    def create(basename: str) -> DuckdbManager:
        """Creates an instance whose in-memory database name is hopefully unique."""
        return DuckdbManager(name=f'{basename}-{DuckdbManager._instance_counter.inc_and_get()}')

    def attach(self, db: DuckdbFilesystemDatabase) -> None:
        with self._lock:
            self._databases[db.name] = db
            self._conn.execute(f"attach database '{db.path}' as {db.name}")

    def detach(self, name: str) -> None:
        with self._lock:
            del self._databases[name]
            self._conn.execute(f'detach database {name}')

    def attached_databases(self) -> list[DuckdbFilesystemDatabase]:
        with self._lock:
            return list(self._databases.values())

    def cursor(self) -> duckdb.DuckDBPyConnection:
        return self._conn.cursor()

    def close(self) -> None:
        self._conn.close()

        
