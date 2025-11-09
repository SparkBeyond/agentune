from __future__ import annotations

import logging
import random
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast, override

import cattrs.strategies
import duckdb
from attr import define
from attrs import field, frozen
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from frozendict import frozendict

import agentune.analyze.core.setup
from agentune.analyze.core import default_duckdb_batch_size
from agentune.analyze.core.schema import Schema
from agentune.analyze.core.types import Dtype
from agentune.analyze.util.attrutil import frozendict_converter
from agentune.analyze.util.cattrutil import UseTypeTag
from agentune.analyze.util.duckdbutil import transaction_scope

if TYPE_CHECKING:
    from agentune.analyze.core.dataset import DatasetSource


_logger = logging.getLogger(__name__)

required_db_version = 'v1.2.0'
'''The minimum required version of duckdb databases, enforced when we create or attach them.

The mapping from duckdb releases to actual format versions is given at 
https://duckdb.org/docs/stable/internals/storage#storage-version-table
Not every released version changes the format, but every released version number is a supported parameter, including 
fix versions like 'v1.2.1'.
'''


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
    
    This represents a table in a real database, not any other relation that DuckDB knows how to read.
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

    def alter_column_types(self, dtypes: Mapping[str, Dtype], conn: DuckDBPyConnection,
                           set_invalid_to_null: bool = False) -> DuckdbTable:
        """Permanently alter the types of columns in the table.

        Implemented using SQL casts; see https://duckdb.org/docs/stable/sql/data_types/typecasting.html.
        If you need custom conversion logic, run SQL statements directly.
        """
        if not dtypes:
            return self

        for name in dtypes:
            if name not in self.schema.names:
                raise ValueError(f'Column {name} not found in table {self.name.name}')

        with conn.cursor() as cursor, transaction_scope(cursor):
            for name, dtype in dtypes.items():
                using_clause = f'USING TRY_CAST("{name}" AS {dtype.duckdb_type})' if set_invalid_to_null else ''
                cursor.execute(f'ALTER TABLE {self.name} ALTER COLUMN "{name}" TYPE {dtype.duckdb_type} {using_clause}')

        return DuckdbTable.from_duckdb(self.name, conn)

    def as_source(self, batch_size: int = default_duckdb_batch_size) -> DatasetSource:
        # Local import to avoid cycle
        from agentune.analyze.core.dataset import DatasetSource
        return DatasetSource.from_table(self, batch_size)

    @staticmethod
    def from_duckdb(name: DuckdbName | str, conn: DuckDBPyConnection) -> DuckdbTable:
        if isinstance(name, str):
            name = DuckdbName.qualify(name, conn)
        return DuckdbTable(name, Schema.from_duckdb(conn.table(str(name))), ArtIndex.from_duckdb(name, conn))

    @staticmethod
    def exists(name: DuckdbName | str, conn: DuckDBPyConnection) -> bool:
        if isinstance(name, str):
            name = DuckdbName.qualify(name, conn)
        conn.execute('SELECT EXISTS(SELECT * FROM duckdb_tables() WHERE database_name = ? AND schema_name = ? AND table_name = ?)',
                    [name.database, name.schema, name.name])
        return cast(tuple[bool], conn.fetchone())[0]


@frozen
class DuckdbIndex(ABC, UseTypeTag):
    """A table index definition.
    
    Make sure to read https://duckdb.org/docs/stable/sql/indexes.html before using.
    """

    @property
    @abstractmethod
    def name(self) -> DuckdbName: ...

    @abstractmethod
    def create(self, conn: DuckDBPyConnection, table_name: DuckdbName | str, if_not_exists: bool = True) -> None: ...

    @abstractmethod
    def drop(self, conn: DuckDBPyConnection) -> None: ...


@frozen
class ArtIndex(DuckdbIndex):
    """The default duckdb index type. See https://duckdb.org/docs/stable/sql/indexes.html

    The duckdb docs warn:
    > ART indexes must currently be able to fit in memory during index creation.
    > Avoid creating ART indexes if the index does not fit in memory during index creation.
    (My understanding is that it's safe to create an index on an empty table before inserting into it.)
    """

    name: DuckdbName
    cols: tuple[str, ...]

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

    @property
    @abstractmethod
    def read_only(self) -> bool: ...

@frozen
class DuckdbInMemory(DuckdbDatabase):
    """Create a new in-memory database.
    
    Does not refer to a named in-memory database in the duckdb sense, i.e. you can't reconnect to it from another connection.

    Passing an instance of this class to DuckdbManager.attach() creates a new database every time;
    you can use the instance to detach that specific database later.
    """
    @property
    @override
    def default_name(self) -> str:
        return 'memory'

    @property
    @override
    def read_only(self) -> bool:
        return False

@frozen
class DuckdbOnDisk(DuckdbDatabase):
    """Open or create a file-backed database."""
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

    Changing settings not declared as attributes of this class can make agentune code not work correctly.
    In particular, these settings are controlled directly and cannot be changed via the config dict:
    - storage_compatibility_version
    - python_enable_replacements

    Params:
        max_memory: The maximum amount of memory to use for both in-memory databases and temporary storage during
                    queries on all databases. When this limit is exceeded, duckdb starts writing to the temp directory.
                    Can be specified in units of bytes (with various prefixes, eg '1GB') or as a percentage of
                    reported system memory (e.g. '80%').
                    The default is 80% of available system RAM.
        temp_dir:   The directory to write to when duckdb runs out of memory. This applies to in-memory databases
                    that don't fit in memory and also to temporary storage during queries.
                    The directory will only be created when duckdb runs out of in-process memory, and it will be deleted
                    on exit.
                    The default is '.tmp' in the current working dir in in-memory mode, or <database_name>.tmp otherwise.
        max_temp_directory_size: The maximum size of the temp directory. When this limit is exceeded, queries fail.
                                 Can be specified in units of bytes (with various prefixes, eg '1GB') or as a percentage of
                                 free disk space (e.g. '80%').
                                 The default is 90% of free disk space.
        threads:    number of (native) threads used by duckdb. The default is the number of CPU cores.
    """
    max_memory: str | None = None
    temp_dir: str | None = None
    max_temp_directory_size: str | None = None
    threads: int | None = None

    config: frozendict[str, str] = field(factory=frozendict, converter=frozendict_converter)

    def _mandatory_config_dict(self) -> dict[str, str]:
        """Config values our code relies on, which should not be changed by users."""
        return {
            'storage_compatibility_version': required_db_version,
            'python_enable_replacements': 'False',
        }

    def to_config_dict(self) -> dict[str, str]:
        set_fields = { k: str(v) for k, v
                       in cattrs.Converter().unstructure_attrs_asdict(self).items()
                       if v is not None and k != 'config' }
        # Note order of precedence
        return {**self.config, **set_fields, **self._mandatory_config_dict()}


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

    The class instance always keeps an open connection instance, so closing all connections outside of this class
    will not free any resources until this class's own .close() is called.
    Code SHOULD scope the use of connection instances obtained from this class, so that resources are freed
    when this class is eventually closed.
    
    There is currently no way to forcibly close a database instance (discard an in-memory database, release files, close threads)
    while live (python) connections remain. We can implement this in the future by using lower-level duckdb APIs.

    The main database (passed to the constructor) is always attached under the default name given to it by duckdb.
    This is the file basename for on-disk databases, and 'memory' for in-memory databases.
    This is a duckdb limitation. Databases attached later can use arbitrary names.

    The following database names are reserved and may NOT be used for either the primary database or later attached ones:

    - 'system', 'temp': reserved by duckdb
    - 'memory': this is the (unchangeable) name used by the primary database, if it's in-memory. To avoid confusion,
                it cannot be used for on-disk databases, or for secondary databases attached later (even if they're in-memory).

    The following schema names are reserved:

    - 'main': the default schema created with every database; code should not try to delete it or (re)create it
    - 'information_schema', 'pg_catalog': reserved by duckdb
    - 'agentune_temp': used by this library for temporary tables and views. This schema is created (empty) when this class
                       connects to a primary database (unless in read-only mode). It is dropped on exit. It is also dropped
                       with all its contents if we discover it exists on start-up (left over from a previous run).
                       This schema is not created in databases attached later.

                       Code should use `DuckdbManager.temp_schema_name`, not the literal name.

    See also docs/using_duckdb.md and docs/duckdb.md.
    """

    _conn: DuckDBPyConnection
    _conn_lock: threading.Lock
    _main_database: DuckdbDatabase
    _databases: dict[str, DuckdbDatabase] # By database name

    _nonce_name_rnd: ClassVar[random.Random] = random.Random()
    # Deliberately no fixed seed to make sure we don't rely on it.
    # This is threadsafe, if a bit slow under concurrent access.

    temp_schema_name: ClassVar[str] = 'agentune_temp'

    def __init__(self, main_database: DuckdbDatabase, config: DuckdbConfig = DuckdbConfig()):
        agentune.analyze.core.setup.setup()

        match main_database:
            case DuckdbInMemory():
                self._conn = duckdb.connect(':memory:', config=config.to_config_dict())
            case DuckdbOnDisk(path, read_only):
                self._conn = duckdb.connect(path, read_only, config=config.to_config_dict())
        self._conn_lock = threading.Lock()
        self._databases = {main_database.default_name: main_database}
        self._main_database = main_database
        # self._conn.load_extension('spatial')

        self._init_temp_schema()


    def temp_random_name(self, basename: str) -> DuckdbName:
        """Return a new, unused name in the temp schema, using the given basename with a random suffix."""
        name =  self.random_name(basename)
        return DuckdbName(name, self._main_database.default_name, DuckdbManager.temp_schema_name)

    @classmethod
    def random_name(cls, basename: str) -> str:
        """Return the basename with a random suffix. Useful for creating temporary schema objects without name conflicts."""
        return f'{basename}_{cls._nonce_name_rnd.randint(1, 10000000000)}'

    def _init_temp_schema(self) -> None:
        with self.cursor() as conn:
            match conn.execute(f"""select count(*) from duckdb_schemas() 
                                   where database_name = '{self._main_database.default_name}'
                                         and schema_name = '{DuckdbManager.temp_schema_name}'""").fetchone():
                case (int(count), ): temp_schema_exists = count > 0
                case other: raise ValueError(f'Unexpected query result {other}')

            if temp_schema_exists:
                if self._main_database.read_only:
                    _logger.warning(f'Temp schema found in database {self._main_database.default_name}, probably left from a previous run. '
                                    f'Cannot drop it in read-only mode. It will be dropped the next time you connect to this database in read-write mode. '
                                    f'Meanwhile, it will keep using up disk space.')
                else:
                    _logger.warning(f'''Temp schema found in database {self._main_database.default_name}. It was probably left from a previous run'''
                                    f'''that didn't exit cleanly. Dropping it.''')
                    conn.execute(f'drop schema "{DuckdbManager.temp_schema_name}" cascade')

            if not self._main_database.read_only:
                conn.execute(f'create schema "{DuckdbManager.temp_schema_name}"')


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
        if isinstance(db, DuckdbOnDisk) and db.read_only:
            options.append('READ_ONLY')
        if isinstance(db, DuckdbOnDisk):
            options.append(f"STORAGE_VERSION '{required_db_version}'")
        options_str = '' if not options else '(' + ', '.join(options) + ')' # empty '()' is invalid
        target = db.path if isinstance(db, DuckdbOnDisk) else ':memory:'
        with self._conn_lock:
            self._conn.execute(f'''ATTACH DATABASE '{target}' AS "{name}" {options_str}''')
            self._databases[name] = db

    def detach(self, name: str) -> None:
        if name == self._main_database.default_name:
            raise ValueError(f'Cannot detach the main database ({name}).')
        with self._conn_lock:
            self._conn.execute(f'DETACH DATABASE "{name}"')
            del self._databases[name]

    def cursor(self) -> duckdb.DuckDBPyConnection:
        """The caller must close the returned cursor at the end of the code scope that uses it.

        The connection instance kept by this class is never exposed to callers; they can only get new cursors via this method.
        The original connection is closed only when the close() method of this class is called.
        """
        with self._conn_lock:
            return self._conn.cursor() # Not sure if .cursor() is threadsafe, better not risk it

    def close(self) -> None:
        with self._conn_lock:
            try:
                if not self._main_database.read_only:
                    self._conn.execute(f'drop schema "{DuckdbManager.temp_schema_name}" cascade')
            except duckdb.ConnectionException as e:
                if 'Connection already closed' not in str(e):
                    raise
            self._conn.close()

    # Convenience methods

    @staticmethod
    def in_memory(config: DuckdbConfig = DuckdbConfig()) -> DuckdbManager:
        return DuckdbManager(DuckdbInMemory(), config)

    @staticmethod
    def on_disk(path: Path, read_only: bool = False,
                config: DuckdbConfig = DuckdbConfig()) -> DuckdbManager:
        return DuckdbManager(DuckdbOnDisk(path, read_only), config)

    def get_table(self, name: DuckdbName) -> DuckdbTable:
        with self.cursor() as conn:
            return DuckdbTable.from_duckdb(name, conn)
    
    def create_table(self, table: DuckdbTable) -> None:
        with self.cursor() as conn:
            table.create(conn)
