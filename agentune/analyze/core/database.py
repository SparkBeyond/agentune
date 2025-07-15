from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import override

from attrs import frozen
from duckdb import DuckDBPyConnection, DuckDBPyRelation

from agentune.analyze.core.schema import Schema


@frozen
class DatabaseTable:
    name: str
    schema: Schema
    indexes: tuple[DatabaseIndex, ...]

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
    def from_duckdb(name: str, conn: DuckDBPyConnection) -> DatabaseTable:
        return DatabaseTable(name, Schema.from_duckdb(conn.table(name)), ArtIndex.from_duckdb(conn, name))

class DatabaseIndex(ABC): 
    """A duckdb index definition.
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
    def from_duckdb(conn: DuckDBPyConnection, name: str) -> Sequence[DatabaseIndex]: ...


@frozen
class ArtIndex(DatabaseIndex):
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
    def from_duckdb(conn: DuckDBPyConnection, name: str) -> tuple[DatabaseIndex, ...]:
        # There's no explicit column in the result specifying the index type, and I haven't found a way to get it.
        # I filter out rtree (spatial) indexes, but if a third type of index shows up, this will report it as an ART index.
        results = conn.execute("SELECT index_name, expressions::VARCHAR[] from duckdb_indexes() "
                               "WHERE table_name = ? AND sql NOT ILIKE '%USING RTREE%'", [name]).fetchmany()
        # duckdb quotes names iff quoting is required; we standardize on unquoted names in python
        return tuple(
            ArtIndex(result[0].strip('"'), tuple(col.strip('"') for col in result[1]))
            for result in results
        )

