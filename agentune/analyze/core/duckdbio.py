"""Code to read and write external data using duckdb, producing DatasetSources and DatasetSinks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal, Self, override

import httpx
import pyarrow as pa
from attrs import frozen
from duckdb import DuckDBPyConnection, DuckDBPyRelation

from agentune.analyze.core import types
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

    @property
    @override
    def schema(self) -> Schema:
        return self.table.schema
    
    @staticmethod
    def from_table(table: str, conn: DuckDBPyConnection) -> DuckdbTableSource:
        """Access the table once to determine its schema."""
        table_with_schema = DuckdbTable.from_duckdb(table, conn)
        return DuckdbTableSource(table_with_schema)
    
    def as_sink(self) -> DuckdbDatasetSink:
        return DuckdbDatasetSink(self.table.name)
    
    @override
    def to_duckdb(self, conn: DuckDBPyConnection) -> DuckDBPyRelation:
        return conn.table(self.table.name)
    
    @override 
    def open(self, conn: DuckDBPyConnection) -> Iterator[Dataset]:
        return duckdb_to_dataset_iterator(self.to_duckdb(conn))

    @override
    def copy_to_thread(self) -> DuckdbTableSource:
        return self 

    @override 
    def to_arrow_reader(self, conn: DuckDBPyConnection) -> pa.RecordBatchReader:
        return self.to_duckdb(conn).fetch_arrow_reader()

    @staticmethod
    def sniff_schema(table_name: str, conn: DuckDBPyConnection) -> DuckdbTableSource:
        """Look at the table once to determine its schema."""
        return DuckdbTableSource(DuckdbTable.from_duckdb(table_name, conn))

type DuckdbDatasetOpener = Callable[[DuckDBPyConnection], DuckDBPyRelation]

@frozen(eq=False, hash=False)
class DatasetSourceFromDuckdb(DatasetSource):
    """Wrap any data that can be read using duckdb as a DatasetSource.
    
    The origin might be local or remote, and might not be queryable at all but only readable in sequence, like a CSV file.
    """
    schema: Schema
    _opener: DuckdbDatasetOpener

    @override 
    def open(self, conn: DuckDBPyConnection) -> Iterator[Dataset]:
        relation = self._opener(conn)
        return duckdb_to_dataset_iterator(relation)

    @override
    def copy_to_thread(self) -> Self:
        return self 

    @override 
    def to_arrow_reader(self, conn: DuckDBPyConnection) -> pa.RecordBatchReader:
        return self._opener(conn).fetch_arrow_reader()

    @override
    def to_duckdb(self, conn: DuckDBPyConnection) -> DuckDBPyRelation:
        return self._opener(conn)

        
# Convenience wrappers. If you want full control over e.g. the parameters to duckdb.read_csv, create an instance directly
# and pass the right `opener` function calling e.g. conn.read_csv with the parameters you want.
# 
# The sniff_xxx methods require a live connection because they need to open the source once to determine its schema
# before returning; this also means they are potentially blocking and need to be run on a sync thread.

def sniff_schema(opener: DuckdbDatasetOpener, conn: DuckDBPyConnection) -> DatasetSourceFromDuckdb:
    """Open the source once to determine its schema."""
    relation = opener(conn)
    schema = Schema.from_duckdb(relation)
    return DatasetSourceFromDuckdb(schema, opener)

def sniff_csv(path: Path | httpx.URL | str, conn: DuckDBPyConnection) -> DatasetSourceFromDuckdb:
    if not isinstance(path, str):
        path = str(path)
    return sniff_schema(lambda conn: conn.read_csv(path), conn)

def ingest(conn: DuckDBPyConnection, table: DuckdbTable | str, data: DatasetSource) -> DuckdbTableSource:
    if isinstance(table, str):
        table = DuckdbTable(table, data.schema)
    sink = DuckdbDatasetSink(table.name)
    sink.write(data, conn)
    return DuckdbTableSource.from_table(table.name, conn)

def ingest_csv(conn: DuckDBPyConnection, table: DuckdbTable | str, path: Path | str) -> DuckdbTableSource:
    return ingest(conn, table, sniff_csv(path, conn))


@frozen 
class DuckdbDatasetSink(DatasetSink):
    """A sink that writes to a duckdb database.
    
    This is a separate class from LocalDuckdbDataset because it doesn't know its schema ahead of time (until data is written).

    NOTE this recreates the target table instead of dropping its contents, removes any indexes,
         and will fail if any other tables' constraints reference it; we might want a different implementation later.
    """
    table_name: str
    
    @override
    def write(self, dataset_source: DatasetSource, conn: DuckDBPyConnection) -> None:
        if isinstance(dataset_source, DuckdbTableSource | DatasetSourceFromDuckdb):
            # replacement scan! Note that this shadows any existing table named input_relation, so we need to 
            # make sure self.table_name isn't named that TODO
            # dataset_source.to_duckdb() is presumed safe to use in the sense that it should preserve all the column types
            with conn.cursor() as cursor:
                input_relation = dataset_source.to_duckdb(cursor)
                cursor.register('input_relation', input_relation)
                cursor.execute(f"CREATE OR REPLACE TABLE '{self.table_name}' AS SELECT * FROM input_relation")
        else:
            # Other sources' implementation of to_duckdb() might lose information about the column types, e.g. by going through an Arrow reader,
            # so we need to explicitly create the table with the right schema
            with conn.cursor() as cursor, dataset_source.to_arrow_reader(cursor) as arrow_reader:
                try:
                    cursor.begin()
                    DuckdbTable(self.table_name, dataset_source.schema).create(cursor, if_not_exists=False, or_replace=True)
                    cursor.register('arrow_reader', arrow_reader)
                    cursor.execute(f"CREATE OR REPLACE TABLE '{self.table_name}' AS SELECT * FROM arrow_reader")
                    cursor.commit()
                except:
                    cursor.rollback()
                    raise


# TODO make EnumDtype extend or provide an enum.Enum so we can refer to these values directly and not as strings
split_column_dtype = types.EnumDtype('train', 'test', 'feature_search', 'feature_eval')

@frozen
class SplitDuckbTable:
    """A table with a split column marking train/test/etc. datasets.
    
    NOTE that rows marked search are included in the train set for most purposes.
    """
    table: DuckdbTable
    split_column_name: str

    def __attrs_post_init__(self) -> None:
        if self.split_column_name not in self.table.schema.names:
            raise ValueError(f'Split column {self.split_column_name} not found in table {self.table.name}')
        if self.table.schema[self.split_column_name].dtype != split_column_dtype:
            raise ValueError(f'Split column {self.split_column_name} has dtype {self.table.schema[self.split_column_name].dtype}, expected {split_column_dtype}')

    @property
    def schema(self) -> Schema:
        return self.table.schema
    
    @property    
    def schema_without_split_column(self) -> Schema:
        return self.table.schema.drop(self.split_column_name)

    def split_category(self, conn: DuckDBPyConnection, category: str,
                       new_subcategory: str, count: float, sample_type: Literal['PERCENT', 'ROWS'], 
                       random_seed: int = 42) -> None:
        """In an existing split table, mark some of a category's rows as a different category.

        Args:
            count: interpreted as a percentage (of the existing rows in the category, not of the whole table)
                   if sample_type is 'PERCENT', or as an absolute number of rows if sample_type is 'ROWS'.

        """
        conn.execute(f'''WITH train_rows(id) AS MATERIALIZED ( -- materialize to make the sampling stable
                            SELECT rowid FROM "{self.table.name}"
                            WHERE "{self.split_column_name}" = $1
                            USING SAMPLE reservoir({count} {sample_type}) REPEATABLE ({random_seed})
                        )
                        UPDATE "{self.table.name}" 
                        SET "{self.split_column_name}" = $2 
                        WHERE rowid IN (SELECT id FROM train_rows)''', [category, new_subcategory])
    
    @staticmethod
    def add_split_column(conn: DuckDBPyConnection, table: DuckdbTable, split_column_name: str,
                         train_fraction: float = 0.8, feature_search_size: int = 10000, 
                         feature_eval_size: int = 100000,
                         random_seed: int = 42) -> SplitDuckbTable:
        """Add a split column to a table:
        1. Mark train_fraction of the rows as train and the rest as test
        2. Out of train, mark feature_search_size rows as feature_search (this is an absolute cap, not a fraction)
        3. Out of remaining (non feature search) train, mark feature_eval_size rows as feature_eval
        """
        if train_fraction <= 0 or train_fraction >= 1:
            raise ValueError(f'train_fraction must be between 0 and 1, got {train_fraction}')
        
        with transaction_scope(conn):
            conn.execute('SELECT setseed(?)', [random_seed / 100])
            conn.execute(f'''ALTER TABLE "{table.name}" ADD COLUMN "{split_column_name}" {split_column_dtype.duckdb_type}
                             DEFAULT CASE WHEN random() < {train_fraction} THEN 'train' ELSE 'test' END
                             ''') # Can't use NOT NULL in ALTER TABLE, have to make the column not null separately
            conn.execute(f'ALTER TABLE "{table.name}" ALTER COLUMN "{split_column_name}" SET NOT NULL')
            conn.execute(f'ALTER TABLE "{table.name}" ALTER COLUMN "{split_column_name}" DROP DEFAULT')
            conn.execute(f'CREATE INDEX "{table.name}_split_index" ON "{table.name}" ("{split_column_name}")') # TODO index naming to avoid collisions
            split_table = SplitDuckbTable(DuckdbTable.from_duckdb(table.name, conn), split_column_name)
            split_table.split_category(conn, 'train', 'feature_search', feature_search_size, 'ROWS', random_seed)
            split_table.split_category(conn, 'train', 'feature_eval', feature_eval_size, 'ROWS', random_seed)
            return split_table
    
    def _split_as_source(self, *split_names: str) -> DatasetSource:
        # Unfortunately conn.execute, the method that does prepared statements, returns a Connection and not a Relation,
        # so I can't use it here without writing more infra code. But the split names are from an enum and don't require escaping,
        # so this isn't particularly dangerous.
        split_names_set = '(' + ', '.join(f"'{name}'" for name in split_names) + ')'
        return DatasetSourceFromDuckdb(self.schema_without_split_column, 
                                       lambda conn: conn.sql(f'SELECT * FROM "{self.table.name}" WHERE "{self.split_column_name}" in {split_names_set}'))
    
    def train(self) -> DatasetSource:
        return self._split_as_source('train', 'feature_search', 'feature_eval')
    
    def test(self) -> DatasetSource:
        return self._split_as_source('test')
    
    def feature_search(self) -> DatasetSource:
        return self._split_as_source('feature_search')

    def feature_eval(self) -> DatasetSource:
        return self._split_as_source('feature_eval')

