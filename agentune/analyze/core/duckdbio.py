"""Code to read and write external data using duckdb, producing DatasetSources and DatasetSinks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Self, override

import pyarrow as pa
from attrs import frozen
from duckdb import CatalogException, DuckDBPyConnection, DuckDBPyRelation

from agentune.analyze.core import default_duckdb_batch_size
from agentune.analyze.core.database import DuckdbName, DuckdbTable
from agentune.analyze.core.dataset import (
    Dataset,
    DatasetSink,
    DatasetSource,
    IfTargetExists,
    duckdb_to_dataset_iterator,
)
from agentune.analyze.core.schema import Schema
from agentune.analyze.util.duckdbutil import transaction_scope


@frozen
class DuckdbTableSource(DatasetSource):
    """A dataset stored in a local duckdb database (possibly an in-memory one).

    This is required as the input type for some operations that can't work on generic DatasetSources.
    """
    table: DuckdbTable
    batch_size: int = default_duckdb_batch_size

    @override
    def cheap_size(self, conn: DuckDBPyConnection) -> int | None:
        return len(self.to_duckdb(conn))

    @property
    @override
    def schema(self) -> Schema:
        return self.table.schema

    def as_sink(self) -> DuckdbTableSink:
        return DuckdbTableSink(self.table.name)
    
    @override
    def to_duckdb(self, conn: DuckDBPyConnection) -> DuckDBPyRelation:
        return conn.table(str(self.table.name))
    
    @override 
    def open(self, conn: DuckDBPyConnection) -> Iterator[Dataset]:
        return duckdb_to_dataset_iterator(self.to_duckdb(conn), batch_size=self.batch_size)

    @override
    def copy_to_thread(self) -> DuckdbTableSource:
        return self 

    @override 
    def to_arrow_reader(self, conn: DuckDBPyConnection) -> pa.RecordBatchReader:
        return self.to_duckdb(conn).fetch_arrow_reader(batch_size=self.batch_size)

    @staticmethod
    def sniff_schema(table_name: DuckdbName, conn: DuckDBPyConnection) -> DuckdbTableSource:
        """Look at the table once to determine its schema."""
        return DuckdbTableSource(DuckdbTable.from_duckdb(table_name, conn))

type DuckdbDatasetOpener = Callable[[DuckDBPyConnection], DuckDBPyRelation]

@frozen(eq=False, hash=False)
class DatasetSourceFromDuckdb(DatasetSource):
    """Wrap any data that can be read using duckdb as a DatasetSource.
    
    The origin might be local or remote, and might not be queryable at all but only readable in sequence, like a CSV file.

    This is opaque and uninspectable. We may want to introduce specific, serializable classes
    like CsvSink with explicit parameters later.
    """
    schema: Schema
    _opener: DuckdbDatasetOpener
    batch_size: int = default_duckdb_batch_size
    allow_cheap_size: bool = False

    def cheap_size(self, conn: DuckDBPyConnection) -> int | None:
        if not self.allow_cheap_size:
            return None
        return len(self._opener(conn))

    @override 
    def open(self, conn: DuckDBPyConnection) -> Iterator[Dataset]:
        relation = self._opener(conn)
        return duckdb_to_dataset_iterator(relation, batch_size=self.batch_size)

    @override
    def copy_to_thread(self) -> Self:
        return self 

    @override 
    def to_arrow_reader(self, conn: DuckDBPyConnection) -> pa.RecordBatchReader:
        return self._opener(conn).fetch_arrow_reader(batch_size=self.batch_size)

    @override
    def to_duckdb(self, conn: DuckDBPyConnection) -> DuckDBPyRelation:
        return self._opener(conn)


def sniff_schema(opener: DuckdbDatasetOpener, conn: DuckDBPyConnection,
                 batch_size: int = default_duckdb_batch_size) -> DatasetSourceFromDuckdb:
    """Open the source once to determine its schema."""
    relation = opener(conn)
    schema = Schema.from_duckdb(relation)
    return DatasetSourceFromDuckdb(schema, opener, batch_size)

def ingest(conn: DuckDBPyConnection, table: DuckdbTable | DuckdbName | str, data: DatasetSource) -> DuckdbTableSource:
    """Copy data into a duckdb table."""
    match table:
        case DuckdbTable():
            table_name = table.name
        case DuckdbName():
            table_name = table
            table = DuckdbTable(table, data.schema)
        case str():
            table_name = DuckdbName.qualify(table, conn)
            table = DuckdbTable(table_name, data.schema)

    sink = DuckdbTableSink(table_name)
    sink.write(data, conn)
    return DuckdbTableSource(table)

@frozen
class DuckdbTableSink(DatasetSink):
    """A sink that writes to a duckdb database.

    When using an existing table (create_table is False), its schema must match the input data.
    When recreating an existing table, the operation will fail if any other objects (such as constraints) reference it.
    Recreating a table does not preserve any indexes that were defined on it.

    Args:
        table_name: table to insert data into
        create_if_not_exists: if the table does not exist, creates it if True, or fails if False.
        if_exists: if the table already exists, whether to replace it (atomically), append to it, or fail.
                   Replacement and appending are both atomic.
    """
    table_name: DuckdbName
    create_if_not_exists: bool = True
    if_exists: IfTargetExists = IfTargetExists.REPLACE

    @override
    def write(self, dataset_source: DatasetSource, conn: DuckDBPyConnection) -> None:
        # conn.register() shadows any existing table named input_relation, so we need to
        # make sure self.table_name isn't named that.
        # dataset_source.to_duckdb() is presumed safe to use in the sense that it should preserve all the column types.
        with conn.cursor() as cursor, transaction_scope(cursor):
            input_relation = dataset_source.to_duckdb(cursor)
            input_relation_name = self.table_name.name + '_input'

            cursor.register(input_relation_name, input_relation)

            try:
                existing_table = DuckdbTable.from_duckdb(self.table_name, cursor)
            except CatalogException as e:
                if 'does not exist' in str(e):
                    existing_table = None
                    if not self.create_if_not_exists:
                        raise ValueError(f'Table does not exist: {self.table_name}') from e
                else:
                    raise

            if existing_table is None or self.if_exists == IfTargetExists.REPLACE:
                cursor.execute(f'CREATE OR REPLACE TABLE {self.table_name} AS SELECT * FROM "{input_relation_name}"')
            elif self.if_exists == IfTargetExists.APPEND:
                if existing_table.schema != dataset_source.schema:
                    raise ValueError(f'Cannot append to table {self.table_name} due to schema mismatch: source schema is {dataset_source.schema},'
                                     f'destination schema is {existing_table.schema}')
                cursor.execute(f'INSERT INTO {self.table_name} SELECT * FROM "{input_relation_name}"')
            else:
                raise ValueError(f'Table already exists: {self.table_name}')

    def as_table(self, conn: DuckDBPyConnection) -> DuckdbTable:
        return DuckdbTable.from_duckdb(self.table_name, conn)

    def as_source(self, conn: DuckDBPyConnection, batch_size: int = default_duckdb_batch_size) -> DuckdbTableSource:
        return DuckdbTableSource(self.as_table(conn), batch_size)


@frozen(eq=False, hash=False)
class DatasetSinkToDuckdb(DatasetSink):
    """Wraps a writer function in a DatasetSink instance.

    This is opaque and uninspectable. We may want to introduce specific, serializable classes
    like CsvSink with explicit parameters later.
    """
    writer: Callable[[DuckDBPyRelation], None]

    @override
    def write(self, dataset_source: DatasetSource, conn: DuckDBPyConnection) -> None:
        relation = dataset_source.to_duckdb(conn)
        self.writer(relation)

