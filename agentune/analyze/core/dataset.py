from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import override

import polars as pl
import pyarrow as pa
from attrs import frozen
from duckdb import DuckDBPyConnection, DuckDBPyRelation

from agentune.analyze.core.schema import Schema, restore_df_types
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

@frozen
class DatasetSourceFromIterable(DatasetSource):
    schema: Schema
    iterable: Iterable[Dataset]

    @override
    def open(self, conn: DuckDBPyConnection) -> Iterator[Dataset]:
        return iter(self.iterable)
    
    @override 
    def to_duckdb(self, conn: DuckDBPyConnection) -> DuckDBPyRelation:
        # TODO need to implement the converse of restore_df_types
        return conn.from_arrow(self.to_arrow_reader(conn))

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
        # TODO need to implement the converse of restore_df_types
        return conn.from_arrow(self.dataset.data.to_arrow())

    @override 
    def to_dataset(self, conn: DuckDBPyConnection) -> Dataset:
        return self.dataset

    @override 
    def copy_to_thread(self) -> DatasetSourceFromDataset:
        return DatasetSourceFromDataset(self.dataset.copy_to_thread())
        
    
class DatasetSink(ABC):
    """Interface for writing data."""

    @abstractmethod
    def write(self, dataset: DatasetSource, conn: DuckDBPyConnection) -> None:
        """Calling again will overwrite the previously written data.
        Calling again while the previous call has not yet completed is undefined.

        A connection is required because some sinks use duckdb to implement writing to them,
        even if the sink is not in a real duckdb database.
        """
        ...

def duckdb_to_dataset_iterator(relation: DuckDBPyRelation) -> Iterator[Dataset]:
    schema = Schema.from_duckdb(relation)
    return iter(Dataset(schema, restore_df_types(pl.DataFrame(batch), schema)) 
                for batch in relation.fetch_arrow_reader())

def duckdb_to_dataset(relation: DuckDBPyRelation) -> Dataset:
    schema = Schema.from_duckdb(relation)
    return Dataset(schema, restore_df_types(relation.pl(), schema))

def duckdb_to_polars(relation: DuckDBPyRelation) -> pl.DataFrame:
    return duckdb_to_dataset(relation).data
