from collections.abc import Iterable, Iterator

import polars as pl
import pyarrow as pa
from attrs import frozen
from duckdb import DuckDBPyRelation

from agentune.analyze.core.dataset import Dataset, DatasetStreamSource
from agentune.analyze.core.schema import Schema
from agentune.analyze.core.types import EnumDtype


def _restore_enum_types(df: pl.DataFrame, schema: Schema) -> pl.DataFrame:
    # Preserve enum types
    for col in schema.cols:
        if isinstance(col.dtype, EnumDtype):
            values = col.dtype.values
            df = df.cast({col.name: pl.Enum(categories=values)})
    return df

@frozen
class _DuckDBToDataframes(Iterable[Dataset]):
    reader: pa.RecordBatchReader
    schema: Schema

    def __iter__(self) -> Iterator[Dataset]:
        for batch in self.reader:
            df = pl.DataFrame(batch)
            df = _restore_enum_types(df, self.schema)
            yield Dataset(self.schema, df)

def duckdb_to_dataset_stream_source(relation: DuckDBPyRelation) -> DatasetStreamSource:
    schema = Schema.from_duckdb(relation)
    iterable = _DuckDBToDataframes(relation.fetch_arrow_reader(), schema)
    return DatasetStreamSource(schema, iterable)

def duckdb_to_dataset(relation: DuckDBPyRelation) -> Dataset:
    schema = Schema.from_duckdb(relation)
    return Dataset(schema, _restore_enum_types(relation.pl(), schema))

def duckdb_to_polars(relation: DuckDBPyRelation) -> pl.DataFrame:
    return duckdb_to_dataset(relation).data
