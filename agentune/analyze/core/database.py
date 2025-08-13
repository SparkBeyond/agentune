from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import override

import cattrs
import duckdb
from attr import define
from attrs import field, frozen
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from frozendict import frozendict

import agentune.analyze.core.setup
from agentune.analyze.core.schema import Schema
from agentune.analyze.util.attrutil import frozendict_converter


@frozen
class DuckdbName:
    """A fully qualified name of a database object (table, index, view, ...) for use in duckdb statements.

    Qualified names are needed to use multiple databases and/or schemas.

    In SQL queries and statements, instances should be stringified and NOT additionally quoted;
    str(DuckdbName) takes care of quoting.

    A fully qualified name's stringification includes all of its components, because we don't know if it will be used
    in a context where the default database or schema is the same as the one explicitly specified here.

    There is currently no implementation of parsing a qualified string into a DuckdbName,
    because we want to encourage the code to use (and construct) DuckdbNames explicitly and not use strings.
    """

    name: str
    database: str
    schema: str = 'main'

    def __str__(self) -> str:
        return f'"{self.database}"."{self.schema}"."{self.name}"'

    @staticmethod
    def qualify(name: str, conn: DuckDBPyConnection) -> DuckdbName:
        """Fill in the current database and schema names from the connection."""
        schema, database = conn.sql('SELECT current_schema(), current_database()').fetchall()[0]
        return DuckdbName(name, database, schema)


@frozen
class DuckdbTable:
    """A table in a DuckDB database.
    
    This represents a table in a real database, not any other relation that  DuckDB knows how to read.
    """
    name: DuckdbName
    schema: Schema
    indexes: tuple[DuckdbIndex, ...] = ()

    def create(self, conn: DuckDBPyConnection, if_not_exists: bool = False, or_replace: bool = False) -> DuckDBPyRelation: 
        if_not = 'IF NOT EXISTS' if if_not_exists else ''
        replace = 'OR REPLACE' if or_replace else ''
        col_specs = [f'"{c.name}" {c.dtype.duckdb_type}' for c in self.schema.cols]
        conn.execute(f'CREATE {replace} TABLE {if_not} {self.name} ({', '.join(col_specs)})')

        existing_index_names = {index.name for index in ArtIndex.from_duckdb(self.name, conn)}
        for index in self.indexes:
            # Running CREATE INDEX IF NOT EXISTS is expensive; even if it already exists, duckdb first builds a new index
            # and then discards it. So we query ourselves to see if it already exists.
            if index.name in existing_index_names:
                if if_not_exists:
                    continue
                if or_replace:
                    # Create index does not support 'or_replace'; we drop and replace it manually in that case
                    index.drop(conn)
            index.create(conn, self.name, if_not_exists)

        return conn.table(str(self.name))

    @staticmethod
    def from_duckdb(name: DuckdbName | str, conn: DuckDBPyConnection) -> DuckdbTable:
        if isinstance(name, str):
            name = DuckdbName.qualify(name, conn)
        return DuckdbTable(name, Schema.from_duckdb(conn.table(str(name))), ArtIndex.from_duckdb(name, conn))


class DuckdbIndex(ABC): 
    """A table index definition.
    
    Make sure to read https://duckdb.org/docs/stable/sql/indexes.html before using.
    """

    # TODO we'll rely on zonemaps a lot for performance of eg time series; 
    #  therefore we need a method for (re)writing a table in the optimal order once we 
    #  know what context definitions we want. (We can of course guess ahead of time and always sort on the datetime
    #  column if there is one, when inserting data, but that's not perfect; sorting is expensive and there might 
    #  be multiple datetime columns.)

    @property
    @abstractmethod
    def name(self) -> DuckdbName: ...

    @abstractmethod
    def create(self, conn: DuckDBPyConnection, table_name: DuckdbName | str, if_not_exists: bool = True) -> None: ...

    @abstractmethod
    def drop(self, conn: DuckDBPyConnection) -> None: ...

    # TODO implement this by returning indexes of all supported types
    #@staticmethod
    #def from_duckdb(name: DuckdbName | str, conn: DuckDBPyConnection) -> Sequence[DuckdbIndex]: ...


@frozen
class ArtIndex(DuckdbIndex):
    name: DuckdbName
    cols: tuple[str, ...]

    # TODO documented warning: ART indexes must currently be able to fit in memory during index creation.
    # Avoid creating ART indexes if the index does not fit in memory during index creation.
    # (in other words, do more sharding??)
    # (I think this doesn't apply if we create an index on an empty table and then insert into it?)

    @override
    def create(self, conn: DuckDBPyConnection, table_name: DuckdbName | str, if_not_exists: bool = True) -> None:
        if isinstance(table_name, str):
            table_name = DuckdbName.qualify(table_name, conn)

        if isinstance(table_name, DuckdbName) and (table_name.database != self.name.database or table_name.schema != self.name.schema):
            raise ValueError(f'Cannot create index {self.name} on table ({table_name}) in a different database or schema')

        # First check if the index already exists; if so, do nothing.
        #  The docs say that IF NOT EXISTS is currently badly implemented; it will spend the time building
        #  the new index anyway, and only then discard it if it already exists.
        if if_not_exists and self.name in {index.name for index in ArtIndex.from_duckdb(table_name, conn)}:
            return
        
        col_specs = ', '.join(f'"{col}"' for col in self.cols)
        # The index name canont be fully qualified in a CREATE INDEX statement.
        # The table name is qualified and that is enough to place the index into the same database and schema as the table.
        conn.execute(f'CREATE INDEX "{self.name.name}" ON {table_name} ({col_specs})')

    @override
    def drop(self, conn: DuckDBPyConnection) -> None:
        conn.execute(f'DROP INDEX {self.name}')

    @staticmethod
    def from_duckdb(table_name: DuckdbName | str, conn: DuckDBPyConnection) -> tuple[DuckdbIndex, ...]:
        if isinstance(table_name, str):
            table_name = DuckdbName.qualify(table_name, conn)

        # There's no explicit column in the result specifying the index type, and I haven't found a way to get it.
        # I filter out rtree (spatial) indexes, but if a third type of index shows up, this will report it as an ART index.
        results = conn.execute("""SELECT index_name, expressions::VARCHAR[] from duckdb_indexes() 
                               WHERE table_name = ? AND database_name = ? AND schema_name = ? 
                               AND sql NOT ILIKE '%USING RTREE%'""",
                               [table_name.name, table_name.database, table_name.schema]).fetchmany()
        # duckdb quotes names in the output of this query iff quoting is required
        return tuple(
            ArtIndex(DuckdbName.qualify(result[0].strip('"'), conn), tuple(col.strip('"') for col in result[1]))
            for result in results
        )



class DuckdbDatabase(ABC):
    @property
    @abstractmethod
    def default_name(self) -> str:
        """The duckdb default for the database name under which the database is attached.
        
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

@frozen
class DuckdbConfig:
    """Connection options supported by DuckdDB.

    They are documented at https://duckdb.org/docs/stable/configuration/overview.html.
    A few are declared here for ease of use; you can pass any additional ones
    in the config dict.

    The attributes defined in this class override keys of the same name placed in the config dict.
    Attributes set to None will let the duckdb default value take effect.

    Changing settings can, of course, make agentune code not work correctly.
    """
    memory_limit: str | None = None # Default is 80% of available system RAM
    threads: int | None = None # Default is the number of CPU cores

    # Settings agentune deliberately modifies from defaults. These are extra likely to break our code
    # if you change them. We do not test agentune with different values of these settings.
    python_enable_replacements: bool | None = False

    kwargs: frozendict[str, object] = field(factory=frozendict, converter=frozendict_converter)

    def to_config_dict(self) -> dict[str, object]:
        set_fields = { k: v for k, v
                       in cattrs.Converter().unstructure_attrs_asdict(self).items()
                       if v is not None and k != 'kwargs' }
        return { **self.kwargs, **set_fields}


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
    to change it locally, but this doesn't affect other connections.
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

    See also docs/duckdb.md for more information.
    """

    _conn: DuckDBPyConnection
    _main_database: DuckdbDatabase
    _databases: dict[str, DuckdbDatabase] # By database name

    def __init__(self, main_database: DuckdbDatabase, config: DuckdbConfig = DuckdbConfig()):
        agentune.analyze.core.setup.setup()

        match main_database:
            case DuckdbInMemoryDatabase():
                self._conn = duckdb.connect(':memory:', config=config.to_config_dict())
            case DuckdbFilesystemDatabase(path, read_only):
                self._conn = duckdb.connect(path, read_only, config=config.to_config_dict())
        self._databases = {main_database.default_name: main_database}
        self._main_database = main_database
        # self._conn.load_extension('spatial')

    def databases(self) -> Mapping[str, DuckdbDatabase]:
        """Return all databases attached to this manager, by database name."""
        return dict(self._databases)

    def attach(self, db: DuckdbDatabase, name: str | None = None) -> None:
        """Attach a database.
        
        Args:
            db: The database to attach.
            name: The name under which to attach the database. If None, the duckdb default is used.
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
        # TODO is a Connection threadsafe enough for .cursor() to be called on it concurrently from parallel threads,
        #  even if that (main) Connection never runs anything else? Or do we need to lock this or use a threadlocal?
        """The caller must close the returned cursor at the end of the code scope that uses it.

        The connection instance kept by this class is never exposed to callers; they can only get new cursors via this method.
        The original connection is closed only when the close() method of this class is called.
        """
        return self._conn.cursor()

    def close(self) -> None:
        self._conn.close()

    # Convenience methods

    @staticmethod
    def in_memory(config: DuckdbConfig = DuckdbConfig()) -> DuckdbManager:
        return DuckdbManager(DuckdbInMemoryDatabase(), config)

    @staticmethod
    def on_disk(path: Path, read_only: bool = False,
                config: DuckdbConfig = DuckdbConfig()) -> DuckdbManager:
        return DuckdbManager(DuckdbFilesystemDatabase(path, read_only), config)

    def get_table(self, name: DuckdbName) -> DuckdbTable:
        with self.cursor() as conn:
            return DuckdbTable.from_duckdb(name, conn)
    
    def create_table(self, table: DuckdbTable) -> None:
        with self.cursor() as conn:
            table.create(conn)
