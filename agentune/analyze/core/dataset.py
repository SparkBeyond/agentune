from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import override

import polars as pl
import pyarrow as pa
from attrs import frozen
from duckdb import DuckDBPyConnection, DuckDBPyRelation

from agentune.analyze.core.schema import Schema
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

    def as_stream(self) -> DatasetStream:
        return DatasetStream(self.schema, iter([self]))
    
    @staticmethod
    def from_polars(df: pl.DataFrame) -> Dataset:
        """Note that some schema information is not represented in a polars DataFrame.
        A schema created from them will have some erased types.
        """
        return Dataset(Schema.from_polars(df), df)

    @override 
    def copy_to_thread(self) -> Dataset:
        return Dataset(self.schema, self.data.clone())

@frozen(eq=False, hash=False)
class DatasetStream(Iterator[Dataset], CopyToThread):
    """A stream of datasets, which can only be read once, and whose schema is known ahead of time."""

    schema: Schema
    iter: Iterator[Dataset]

    @override
    def copy_to_thread(self) -> DatasetStream:
        return DatasetStream(self.schema, iter(dataset.copy_to_thread() for dataset in self.iter)) 

    def __iter__(self) -> Iterator[Dataset]: 
        return self.iter
    
    def __next__(self) -> Dataset:
        return next(self.iter)
    
    def to_arrow_reader(self) -> pa.RecordBatchReader:
        return pa.RecordBatchReader.from_batches(self.schema.to_arrow(), 
                                                 itertools.chain.from_iterable(dataset.data.to_arrow().to_batches() for dataset in self))
    
    def to_duckdb(self, conn: DuckDBPyConnection) -> DuckDBPyRelation:
        return conn.from_arrow(self.to_arrow_reader())

@frozen(eq=False, hash=False) # The 'iterable' may not be comparable or hashable
class DatasetStreamSource(Iterable[Dataset], CopyToThread):
    """A source of a dataset stream which can be read multiple times, and whose schema is known ahead of time."""

    schema: Schema
    iterable: Iterable[Dataset]

    @override
    def copy_to_thread(self) -> DatasetStreamSource:
        return DatasetStreamSource(self.schema, (dataset.copy_to_thread() for dataset in self.iterable))

    def open(self) -> DatasetStream:
        return DatasetStream(self.schema, iter(self.iterable))

    def __iter__(self) -> Iterator[Dataset]: 
        return iter(self.iterable)

class DatasetSink(ABC):
    """Interface for telling pipelines where to write their outputs."""

    @abstractmethod
    def write(self, dataset: DatasetStream) -> None:
        """Calling again will overwrite the previously written data.
        Calling again while the previous call has not yet completed is undefined.
        This applies to calling async_write too.
        """
        ...

    @abstractmethod
    async def async_write(self, dataset: DatasetStream) -> None:
        """Does not block the calling thread. Only guarantees a best-attempt at actually being async on the thread(s)
        it actually runs on.
        """
        ...
