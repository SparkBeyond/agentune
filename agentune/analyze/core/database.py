from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import override

import duckdb
from attr import define
from attrs import frozen
from duckdb import DuckDBPyConnection, DuckDBPyRelation

import agentune.analyze.core.setup
from agentune.analyze.core.schema import Schema
from agentune.analyze.util.atomic import AtomicInt


@frozen
class DuckdbTable:
    """A table in a DuckDB database.
    
    This represents a table in a real database, not any other relation that  DuckDB knows how to read.
    """
    # TODO add catalog/schema names
    name: str
    schema: Schema
    indexes: tuple[DuckdbIndex, ...] = ()

    def create(self, conn: DuckDBPyConnection, if_not_exists: bool = False, or_replace: bool = False) -> DuckDBPyRelation: 
        if_not = 'IF NOT EXISTS' if if_not_exists else ''
        replace = 'OR REPLACE' if or_replace else ''
        col_specs = [f'"{c.name}" {c.dtype.duckdb_type}' for c in self.schema.cols]
        conn.execute(f'CREATE {replace} TABLE {if_not} "{self.name}" ({', '.join(col_specs)})')

        existing_index_names = {index.name for index in ArtIndex.from_duckdb(conn, self.name)}
        for index in self.indexes:
            # Running CREATE INDEX IF NOT EXISTS is expensive; even if it already exists, duckdb first builds a new index
            # and then discards it. So we query ourselves to see if it already exists.
            if index.name in existing_index_names:
                if if_not_exists:
                    continue
                if or_replace:
                    # Create index does not support 'or_replace'; we drop and replace it manually in that case
                    conn.execute(f'DROP INDEX "{index.name}"')
            index.create(conn, self.name, if_not_exists)

        return conn.table(self.name)

    @staticmethod
    def from_duckdb(name: str, conn: DuckDBPyConnection) -> DuckdbTable:
        return DuckdbTable(name, Schema.from_duckdb(conn.table(name)), ArtIndex.from_duckdb(conn, name))

class DuckdbIndex(ABC): 
    """A table index definition.
    
    Make sure to read https://duckdb.org/docs/stable/sql/indexes.html before using.
    """

    # TODO we'll rely on zonemaps a lot for performance of eg time series; 
    #  therefore we need a method for (re)writing a table in the optimal order once we 
    #  know what context objects we want. (We can of course guess ahead of time and always sort on the datetime
    #  column if there is one, when inserting data, but that's not perfect; sorting is expensive and there might 
    #  be multiple datetime columns.)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def create(self, conn: DuckDBPyConnection, table_name: str, if_not_exists: bool = True) -> None: ...

    # TODO specify the database, schema and table name separately.
    #  In the query implementation, we have to pass those three separately.
    #  (Requiring a fully qualified name to be passed in isn't any simpler.)
    @staticmethod
    @abstractmethod
    def from_duckdb(conn: DuckDBPyConnection, name: str) -> Sequence[DuckdbIndex]: ...


@frozen
class ArtIndex(DuckdbIndex):
    name: str
    cols: tuple[str, ...]

    # TODO documented warning: ART indexes must currently be able to fit in memory during index creation.
    # Avoid creating ART indexes if the index does not fit in memory during index creation.
    # (in other words, do more sharding??)
    # (I think this doesn't apply if we create an index on an empty table and then insert into it?)

    @override
    def create(self, conn: DuckDBPyConnection, table_name: str, if_not_exists: bool = True) -> None:
        # First check if the index already exists; if so, do nothing.
        #  The docs say that IF NOT EXISTS is currently badly implemented; it will spend the time building
        #  the new index anyway, and only then discard it if it already exists.
        if if_not_exists and self.name in {index.name for index in ArtIndex.from_duckdb(conn, table_name)}:
            return
        
        col_specs = ', '.join(f'"{col}"' for col in self.cols)
        conn.execute(f'CREATE INDEX "{self.name}" ON "{table_name}" ({col_specs})')

    @override
    @staticmethod
    def from_duckdb(conn: DuckDBPyConnection, name: str) -> tuple[DuckdbIndex, ...]:
        # There's no explicit column in the result specifying the index type, and I haven't found a way to get it.
        # I filter out rtree (spatial) indexes, but if a third type of index shows up, this will report it as an ART index.
        results = conn.execute("SELECT index_name, expressions::VARCHAR[] from duckdb_indexes() "
                               "WHERE table_name = ? AND sql NOT ILIKE '%USING RTREE%'", [name]).fetchmany()
        # duckdb quotes names iff quoting is required; we standardize on unquoted names in python
        return tuple(
            ArtIndex(result[0].strip('"'), tuple(col.strip('"') for col in result[1]))
            for result in results
        )


@define
class DuckdbDatabase(ABC):
    name: str # Name to attach as
    read_only: bool

@frozen
class DuckdbInMemoryDatabase(DuckdbDatabase):
    pass

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
    
    def in_memory_database(self, read_only: bool = False) -> DuckdbInMemoryDatabase:
        return DuckdbInMemoryDatabase(self.name, read_only)

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

