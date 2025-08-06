from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from io import StringIO, TextIOBase
from pathlib import Path
from typing import Any, override

import httpx
import polars as pl
import pyarrow as pa
from attrs import frozen
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from tests.agentune.analyze.core import default_duckdb_batch_size

from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.schema import Schema, restore_df_types, restore_relation_types
from agentune.analyze.core.threading import CopyToThread
from agentune.analyze.util.polarutil import df_field


@frozen
class Dataset(CopyToThread):
    """A dataframe with a schema.
    
    This class exists because our Schema might evolve to contain more information than the pl.Schema available on the dataframe.
    """

    schema: Schema
    data: pl.DataFrame = df_field()

    def drop(self, *names: str) -> Dataset:
        """Drop columns."""
        return Dataset(self.schema.drop(*names), self.data.drop(*names))
    
    def head(self, n: int) -> Dataset:
        """Get the first n rows."""
        return Dataset(self.schema, self.data.head(n))
    
    def tail(self, n: int) -> Dataset:
        """Get the last n rows."""
        return Dataset(self.schema, self.data.tail(n))
    
    def skip(self, n: int) -> Dataset:
        """Skip the first n rows."""
        return Dataset(self.schema, self.data.slice(n))
    
    def slice(self, offset: int, length: int | None) -> Dataset:
        return Dataset(self.schema, self.data.slice(offset, length))

    def hstack(self, other: Dataset) -> Dataset:
        return Dataset(self.schema.hstack(other.schema), self.data.hstack(other.data))

    def vstack(self, other: pl.DataFrame | Dataset) -> Dataset:
        if isinstance(other, Dataset):
            if other.schema != self.schema:
                raise ValueError('Cannot vstack, schema mismatch')
            other = other.data
        elif self.schema.to_polars() != other.schema:
            raise ValueError('Cannot vstack, schema mismatch')
        return Dataset(self.schema, self.data.vstack(other))

    def select(self, *cols: str) -> Dataset:
        return Dataset(self.schema.select(*cols), self.data.select(cols))
    
    @property
    def height(self) -> int:
        return self.data.height
    
    def __len__(self) -> int:
        return self.height
    
    def empty(self) -> Dataset:
        return Dataset(self.schema, self.data.clear())

    @staticmethod
    def from_polars(df: pl.DataFrame) -> Dataset:
        """Note that some schema information is not represented in a polars DataFrame.
        A schema created from them will have some erased types.
        """
        return Dataset(Schema.from_polars(df), df)

    # TODO method from_pandas

    def as_source(self) -> DatasetSource:
        return DatasetSourceFromDataset(self)

    @override 
    def copy_to_thread(self) -> Dataset:
        return Dataset(self.schema, self.data.clone())


class DatasetSource(CopyToThread):
    """A source of a dataset stream which can be read multiple times, and whose schema is known ahead of time."""

    @property
    @abstractmethod
    def schema(self) -> Schema: ...

    @abstractmethod
    def open(self, conn: DuckDBPyConnection) -> Iterator[Dataset]: ...

    def to_arrow_reader(self, conn: DuckDBPyConnection) -> pa.RecordBatchReader:
        return pa.RecordBatchReader.from_batches(self.schema.to_arrow(), 
                                                 itertools.chain.from_iterable(dataset.data.to_arrow().to_batches() for dataset in self.open(conn)))
    @abstractmethod
    def to_duckdb(self, conn: DuckDBPyConnection) -> DuckDBPyRelation: ...
    
    def to_dataset(self, conn: DuckDBPyConnection) -> Dataset:
        """Read the entire source into memory."""
        return Dataset(self.schema, self.to_duckdb(conn).pl())

    @staticmethod
    def from_dataset(dataset: Dataset) -> DatasetSourceFromDataset:
        return DatasetSourceFromDataset(dataset)

    @staticmethod
    def from_datasets(schema: Schema, datasets: Iterable[Dataset]) -> DatasetSourceFromIterable:
        return DatasetSourceFromIterable(schema, datasets)

    @staticmethod
    def from_table(table: DuckdbTable, batch_size: int = default_duckdb_batch_size) -> DatasetSource:
        # Local import to avoid circle
        from agentune.analyze.core.duckdbio import DuckdbTableSource
        return DuckdbTableSource(table, batch_size)

    @staticmethod
    def from_table_name(table_name: str, conn: DuckDBPyConnection, batch_size: int = default_duckdb_batch_size) -> DatasetSource:
        return DatasetSource.from_table(DuckdbTable.from_duckdb(table_name, conn), batch_size)

    @staticmethod
    def from_duckdb_parser(opener: Callable[[DuckDBPyConnection], DuckDBPyRelation],
                           conn_or_schema: DuckDBPyConnection | Schema) -> DatasetSource:
        """Read any data that duckdb can access by supplying an explicit query or method call on a Connection.

        Args:
             conn_or_schema: if the Schema is known, a DatasetSource is returned immediately.
                             Otherwise, a connection must be given, and the dataset will be opened once
                             (but not read fully) in order to find out its schema.
        """
        from agentune.analyze.core.duckdbio import DatasetSourceFromDuckdb, sniff_schema
        if isinstance(conn_or_schema, DuckDBPyConnection):
            return sniff_schema(opener, conn_or_schema)
        else:
            return DatasetSourceFromDuckdb(conn_or_schema, opener)

    @staticmethod
    def from_csv(path: Path | httpx.URL | str | StringIO | TextIOBase, conn: DuckDBPyConnection,
                 **kwargs: Any) -> DatasetSource:
        """Read CSV, from a local path or remote URL or from in-memory data or a stream.

        CSV reading is implemented by duckdb and is highly configurable. You can read about the
        possible arguments (that go in the **kwargs) at https://duckdb.org/docs/stable/data/csv/overview.html,
        and see the python signature in `duckdb.read_csv`.
        """
        from agentune.analyze.core.duckdbio import sniff_schema
        if isinstance(path, Path | httpx.URL):
            path = str(path)
        return sniff_schema(lambda conn: conn.read_csv(path, **kwargs), conn)

    @staticmethod
    def from_parquet(path: Path | httpx.URL | str, conn: DuckDBPyConnection,
                     **kwargs: Any) -> DatasetSource:
        """Read CSV, from a local path or remote URL or from in-memory data or a stream.

        CSV reading is implemented by duckdb and is configurable. You can read about the
        possible arguments (that go in the **kwargs) at https://duckdb.org/docs/stable/data/parquet/overview.html,
        and see the python signature in `duckdb.read_parquet`.
        """
        from agentune.analyze.core.duckdbio import sniff_schema
        if isinstance(path, Path | httpx.URL):
            path = str(path)
        return sniff_schema(lambda conn: conn.read_parquet(path, **kwargs), conn)

    @staticmethod
    def from_json(path: Path | httpx.URL | str, conn: DuckDBPyConnection, **kwargs: Any) -> DatasetSource:
        """Read json, from a local path or remote URL or from in-memory data or a stream.

        Json reading is implemented by duckdb and is configurable. You can read about the
        possible arguments (that go in the **kwargs) at https://duckdb.org/docs/stable/data/json/overview.html,
        and see the python signature in `duckdb.read_json`.
        """
        from agentune.analyze.core.duckdbio import sniff_schema
        if isinstance(path, Path | httpx.URL):
            path = str(path)
        return sniff_schema(lambda conn: conn.read_json(path, **kwargs), conn)


@frozen
class DatasetSourceFromIterable(DatasetSource):
    schema: Schema
    iterable: Iterable[Dataset]

    @override
    def open(self, conn: DuckDBPyConnection) -> Iterator[Dataset]:
        return iter(self.iterable)
    
    @override 
    def to_duckdb(self, conn: DuckDBPyConnection) -> DuckDBPyRelation:
        return restore_relation_types(conn.from_arrow(self.to_arrow_reader(conn)), self.schema)

    @override
    def to_dataset(self, conn: DuckDBPyConnection) -> Dataset:
        iterator = iter(self.iterable)
        df = next(iterator).data
        for more in iterator:
            df = df.vstack(more.data, in_place=True)
        return Dataset(self.schema, df)

    @override 
    def copy_to_thread(self) -> DatasetSourceFromIterable:
        return DatasetSourceFromIterable(self.schema, (dataset.copy_to_thread() for dataset in self.iterable))

@frozen
class DatasetSourceFromDataset(DatasetSource):
    dataset: Dataset

    @property
    @override
    def schema(self) -> Schema:
        return self.dataset.schema
    
    @override
    def open(self, conn: DuckDBPyConnection) -> Iterator[Dataset]:
        return iter([self.dataset])
    
    @override
    def to_duckdb(self, conn: DuckDBPyConnection) -> DuckDBPyRelation:
        return restore_relation_types(conn.from_arrow(self.dataset.data.to_arrow()), self.schema)

    @override 
    def to_dataset(self, conn: DuckDBPyConnection) -> Dataset:
        return self.dataset

    @override 
    def copy_to_thread(self) -> DatasetSourceFromDataset:
        return DatasetSourceFromDataset(self.dataset.copy_to_thread())

    
class DatasetSink(ABC):
    """Interface for writing data.

    Note that this interface does not know the expected schema.
    Depending on the implementation, it might be able to write data with any schema,
    or only with a specific one.
    """

    @abstractmethod
    def write(self, dataset: DatasetSource, conn: DuckDBPyConnection) -> None:
        """Calling again will overwrite the previously written data.
        Calling again while the previous call has not yet completed is undefined.

        A connection is required because some sinks use duckdb to implement writing to them,
        even if the sink is not in a real duckdb database.
        """
        ...

    @staticmethod
    def into_duckdb_table(table_name: str, create_table: bool = True,
                          or_replace: bool = True, delete_contents: bool = True) -> DatasetSink:
        """See DuckdbDatasetSink for the arguments."""
        # Local import to avoid circle
        from agentune.analyze.core.duckdbio import DuckdbTableSink
        return DuckdbTableSink(table_name, create_table, or_replace, delete_contents)

    @staticmethod
    def into_duckdb(writer: Callable[[DuckDBPyRelation], None]) -> DatasetSink:
        """Wrap a custom function that takes a Relation and saves it somewhere."""
        from agentune.analyze.core.duckdbio import DatasetSinkToDuckdb
        return DatasetSinkToDuckdb(writer)

    @staticmethod
    def into_csv(path: Path | str, **kwargs: Any) -> DatasetSink:
        """Write to a CSV local file or files.

        CSV writing is implemented by duckdb and is highly configurable. You can read about the
        possible arguments (that go in the **kwargs) at
        https://duckdb.org/docs/stable/sql/statements/copy.html#csv-options,
        and the Python API of `duckdb.Connection.write_csv`.
        """
        from agentune.analyze.core.duckdbio import DatasetSinkToDuckdb
        if isinstance(path, Path):
            path = str(path)
        return DatasetSinkToDuckdb(lambda relation: relation.write_csv(path, **kwargs))

    @staticmethod
    def into_parquet(path: Path | str, **kwargs: Any) -> DatasetSink:
        """Write to a Parquet local file or files.

        Parquet writing is implemented by duckdb and is configurable. You can read about the
        possible arguments (that go in the **kwargs) at
        https://duckdb.org/docs/stable/sql/statements/copy.html#parquet-options,
        and the Python API of `duckdb.Connection.write_parquet`.
        """
        from agentune.analyze.core.duckdbio import DatasetSinkToDuckdb
        if isinstance(path, Path):
            path = str(path)
        return DatasetSinkToDuckdb(lambda relation: relation.write_parquet(path, **kwargs))


def duckdb_to_dataset_iterator(relation: DuckDBPyRelation, batch_size: int = 10000) -> Iterator[Dataset]:
    schema = Schema.from_duckdb(relation)
    return iter(Dataset(schema, restore_df_types(pl.DataFrame(batch), schema)) 
                for batch in relation.fetch_arrow_reader(batch_size=batch_size))

def duckdb_to_dataset(relation: DuckDBPyRelation) -> Dataset:
    schema = Schema.from_duckdb(relation)
    return Dataset(schema, restore_df_types(relation.pl(), schema))

def duckdb_to_polars(relation: DuckDBPyRelation) -> pl.DataFrame:
    return duckdb_to_dataset(relation).data
