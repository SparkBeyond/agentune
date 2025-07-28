from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import override

import duckdb
from attr import define
from attrs import frozen
from duckdb import DuckDBPyConnection, DuckDBPyRelation

import agentune.analyze.core.setup
from agentune.analyze.core.schema import Schema


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


class DuckdbDatabase(ABC):
    @property
    @abstractmethod
    def default_name(self) -> str:
        """The duckdb default for the catalog name under which the database is attached.
        
        This is the name used for the first database opened by a DuckdbManager instance;
        databases attached later can override the name used.
        """
        ...

@frozen
class DuckdbInMemoryDatabase(DuckdbDatabase):
    """An in-memory database. 
    
    Not a named in-memory database in the duckdb sense, i.e. you can't reconnect to it from another connection.

    Passing an instance of this class to DuckdbManager.attach() creates a new database every time;
    you can keep the instance in order to detach that specific database later.
    """
    @property
    @override
    def default_name(self) -> str:
        return 'memory'

@frozen
class DuckdbFilesystemDatabase(DuckdbDatabase):
    """A file-backed database we can create or open."""
    path: Path
    read_only: bool = False

    @property
    @override
    def default_name(self) -> str:
        return self.path.stem

@define(init=False, eq=False, hash=False)
class DuckdbManager:
    """Manages duckdb databases and connections, and relatedly, the size and creation of the duckbd threadpool(s).

    This class is NOT thread-safe while you're attaching or detaching databases.
    Afterwards, connection instances you acquire from the cursor() method are also not thread-safe,
    and every thread needs to acquire its own connection. You may also wish to create cursors
    to separate connection-level effects like transactions and USE statements.

    Each instance of this class starts out by connecting to one database and can attach more databases later.
    A single threadpool is used, no matter how many databases are attached.
    Attaching and detaching databases affects all connection instances previously returned by cursor().
        
    The default database (out of those attached) is always the first one; you can execute USE on a connection
    to change it locally but this doesn't affect other connections.
    (This limitation is required because otherwise all code calling connection.cursor() would have to re-apply
    the USE statement, and most code calls connection.cursor() and not the cursor() method of this class because
    it doesn't have a reference to this class.)

    The class instance always keeps an open connection instance, so closing all connections outside of this class
    will not free any resources until this class's own .close() is called.
    Code SHOULD scope the use of connection instances obtained from this class, so that resources are freed
    when this class is eventually closed.
    
    There is currently no way to force close a database instance (discard an in-memory database, release files, close threads) 
    while live (python) connections remain. We can implement this in the future by using lower-level duckdb APIs.

    The main database (passed to the constructor) is always attached under the default name given to it by duckdb.
    This is the file basename for on-disk databases, and 'memory' for in-memory databases.
    This is a duckdb limitation. Databases attached later can use arbitrary names.

    -----------------------

    Notes on duckdb's own behavior (TODO try to confirm yet again that I'm right, find some references):

    1. Every call to duckdb.connect(path) for a new database path creates a new 'database instance', which includes some 
       per-DB-instance settings and state, and a threadpool (whose size can then be changed).
       
       Calling duckdb.connect(path) for a path that already has a live connection in the process returns 
       a new connection to the same database instance; this works until all connections to the database instance
       are explicitly closed or are garbage collected.

    2. Once you have a connection, you can call its .cursor() method to get another connection to the same DB instance.
       This is cheap.

       Connections are blocking, so at minimum we need a cursor per thread. Also, some things happen at connection 
       scope, and it's useful to create connections to scope various effects, like USE statements.

    3. More databases can be attached to an existing connection; this does not create an additional threadpool.
       
       Attaching/detaching databases affects existing connections created via cursor() calls from each other;
       only connections created by calling duckdb.connect() are unaffected. (Such connections can still share
       the database instance.)

    4. It's possible to reconnect to an in-memory database, as long as it exists, by calling duckdb.connect(f':memory:{name}').
       However, it's not possible to attach an existing in-memory database to a connection; only new 'anonymous'
       in-memory databases can be attached.

    4. Each database (=catalog) can then have multiple schemas, as normal in SQL. 
       See https://duckdb.org/docs/stable/sql/statements/attach#name-qualification
    """

    _conn: DuckDBPyConnection
    _main_database: DuckdbDatabase
    _databases: dict[str, DuckdbDatabase] # By catalog name

    def __init__(self, main_database: DuckdbDatabase):
        agentune.analyze.core.setup.setup()

        match main_database:
            case DuckdbInMemoryDatabase():
                self._conn = duckdb.connect(':memory:')
            case DuckdbFilesystemDatabase(path, read_only):
                self._conn = duckdb.connect(path, read_only)
        self._databases = {main_database.default_name: main_database} 
        self._main_database = main_database

        # TODO disable this, and the INSTALL in setup.py, while we don't have any code that uses the spatial extension
        self._conn.load_extension('spatial')

    def databases(self) -> Mapping[str, DuckdbDatabase]:
        """Return all databases attached to this manager, by catalog name."""
        return dict(self._databases)

    def attach(self, db: DuckdbDatabase, name: str | None = None) -> None:
        """Attach a database.
        
        Args:
            db: The database to attach.
            name: The catalog name under which to attach the database. If None, the duckdb default is used.
        """
        if name is None:
            name = db.default_name
        if name in self._databases:
            raise ValueError(f'A database with the same name ({name}) is already attached.')
        
        options = []
        if isinstance(db, DuckdbFilesystemDatabase) and db.read_only:
            options.append('READ_ONLY')
        if isinstance(db, DuckdbFilesystemDatabase):
            options.append("STORAGE_VERSION 'v1.2.0'")
        options_str = '' if not options else '(' + ', '.join(options) + ')' # empty '()' is invalid
        target = db.path if isinstance(db, DuckdbFilesystemDatabase) else ':memory:'
        self._conn.execute(f'''ATTACH DATABASE '{target}' AS "{name}" {options_str}''')

        self._databases[name] = db

    def detach(self, name: str) -> None:
        if name == self._main_database.default_name:
            raise ValueError(f'Cannot detach the main database ({name}).')
        self._conn.execute(f'DETACH DATABASE "{name}"')
        del self._databases[name]

    def cursor(self) -> duckdb.DuckDBPyConnection:
        return self._conn.cursor()

    def close(self) -> None:
        self._conn.close()

    # Convenience methods

    @staticmethod
    def in_memory() -> DuckdbManager:
        return DuckdbManager(DuckdbInMemoryDatabase())

    @staticmethod
    def on_disk(path: Path, read_only: bool = False) -> DuckdbManager:
        return DuckdbManager(DuckdbFilesystemDatabase( path, read_only))

    def get_table(self, name: str) -> DuckdbTable:
        with self.cursor() as conn:
            return DuckdbTable.from_duckdb(name, conn)
    
    def create_table(self, table: DuckdbTable) -> None:
        with self.cursor() as conn:
            table.create(conn)
