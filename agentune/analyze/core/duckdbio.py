"""Code to read and write external data using duckdb, producing DatasetSources and DatasetSinks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Self, override

import pyarrow as pa
from attrs import frozen
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from tests.agentune.analyze.core import default_duckdb_batch_size

from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.dataset import (
    Dataset,
    DatasetSink,
    DatasetSource,
    duckdb_to_dataset_iterator,
)
from agentune.analyze.core.schema import Schema
from agentune.analyze.util.duckdbutil import transaction_scope


@frozen
class DuckdbTableSource(DatasetSource):
    """A dataset stored in a local duckdb database (possibly an in-memory one).

    This is required as the input type for some operations that can't work on generic DatasetSources, most often context data.
    """
    table: DuckdbTable
    batch_size: int = default_duckdb_batch_size

    @property
    @override
    def schema(self) -> Schema:
        return self.table.schema

    def as_sink(self) -> DuckdbTableSink:
        return DuckdbTableSink(self.table.name)
    
    @override
    def to_duckdb(self, conn: DuckDBPyConnection) -> DuckDBPyRelation:
        return conn.table(self.table.name)
    
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
    def sniff_schema(table_name: str, conn: DuckDBPyConnection) -> DuckdbTableSource:
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


def sniff_schema(opener: DuckdbDatasetOpener, conn: DuckDBPyConnection) -> DatasetSourceFromDuckdb:
    """Open the source once to determine its schema."""
    relation = opener(conn)
    schema = Schema.from_duckdb(relation)
    return DatasetSourceFromDuckdb(schema, opener)

def ingest(conn: DuckDBPyConnection, table: DuckdbTable | str, data: DatasetSource) -> DuckdbTableSource:
    if isinstance(table, str):
        table = DuckdbTable(table, data.schema)
    sink = DuckdbTableSink(table.name)
    sink.write(data, conn)
    return DuckdbTableSource(table)

def ingest_csv(conn: DuckDBPyConnection, table: DuckdbTable | str, path: Path | str) -> DuckdbTableSource:
    return ingest(conn, table, DatasetSource.from_csv(path, conn))


@frozen
class DuckdbTableSink(DatasetSink):
    """A sink that writes to a duckdb database.

    When using an existing table (create_table is False), its schema must match the input data.
    When recreating an existing table, the operation will fail if any other catalog objects reference it.
    Recreating a table does not preserve any indexes that were defined on it.

    Args:
        table_name: table to insert data into
        create_table: if True, creates the target table.
                      This will fail if it already exists and or_replace is False.
        or_replace: when creating the table, replace any existing table.
                    Has no effect if create_table is True.
        delete_contents: when writing to an existing table, delete all its contents first.
                         If False, the new data will be appended instead.
                         Has no effect if create_table is True.
    """
    table_name: str
    create_table: bool = True
    or_replace: bool = True
    delete_contents: bool = True

    @override
    def write(self, dataset_source: DatasetSource, conn: DuckDBPyConnection) -> None:
        # replacement scan! Note that this shadows any existing table named input_relation, so we need to
        # make sure self.table_name isn't named that TODO nonce names
        # dataset_source.to_duckdb() is presumed safe to use in the sense that it should preserve all the column types
        with conn.cursor() as cursor, transaction_scope(cursor):
            input_relation = dataset_source.to_duckdb(cursor)
            cursor.register('input_relation', input_relation)

            if self.create_table:
                replace = 'OR REPLACE' if self.or_replace else ''
                cursor.execute(f"CREATE {replace} TABLE '{self.table_name}' AS SELECT * FROM input_relation")
            else:
                existing_table = DuckdbTable.from_duckdb(self.table_name, cursor)
                if existing_table.schema != dataset_source.schema:
                    raise ValueError(f'Cannot write data with schema {dataset_source.schema} to table {self.table_name} '
                                     f'which has the schema {existing_table.schema}')
                if self.delete_contents:
                    cursor.execute(f"DELETE FROM '{self.table_name}'")
                cursor.execute(f"INSERT INTO '{self.table_name}' SELECT * FROM input_relation")


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

