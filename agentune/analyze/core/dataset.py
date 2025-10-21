from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

from agentune.analyze.core.types import Dtype

if TYPE_CHECKING:
    from agentune.analyze.core.duckdbio import DuckdbTableSink

import httpx
import polars as pl
import pyarrow as pa
from attrs import define, frozen
from duckdb import DuckDBPyConnection, DuckDBPyRelation

from agentune.analyze.core import default_duckdb_batch_size
from agentune.analyze.core.database import DuckdbName, DuckdbTable
from agentune.analyze.core.schema import Field, Schema, restore_df_types, restore_relation_types
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

    def cast(self, dtypes: Mapping[str | Field, Dtype], strict: bool = True) -> Dataset:
        """Change the type of some columns.

        Casting is done using polars; see polars.DataFrame.cast.
        For more complex transformations, manipulate the polars DataFrame directly.

        Args:
            strict: if True, values that fail to be converted raise an error. If False, null (the missing value) is substituted.
        """
        if not dtypes:
            return self

        names_to_dtypes = {
            key.name if isinstance(key, Field) else key: dtype
            for key, dtype in dtypes.items()
        }
        for name in names_to_dtypes:
            if name not in self.schema.names:
                raise ValueError(f'Column {name} not in schema')
        new_schema = Schema(tuple( Field(field.name, names_to_dtypes[field.name]) if field.name in names_to_dtypes else field
                                   for field in self.schema.cols))
        new_data = self.data.cast({name: dtype.polars_type for name, dtype in names_to_dtypes.items()}, strict=strict)
        return Dataset(new_schema, new_data)


    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dataset):
            return False
        return self.schema == other.schema and self.data.equals(other.data)

    @staticmethod
    def from_polars(df: pl.DataFrame) -> Dataset:
        """Note that some schema information is not represented in a polars DataFrame.
        A schema created from them will have some erased types.
        """
        return Dataset(Schema.from_polars(df), df)

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

    def cheap_size(self, _conn: DuckDBPyConnection) -> int | None:
        """Returns the size if it can be checked cheaply, or None otherwise.

        May require a small amount of IO, but not reading the whole input.
        """
        return None # The default

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

    def select(self, *cols: str, batch_size: int = default_duckdb_batch_size) -> DatasetSource:
        new_schema = self.schema.select(*cols)
        def opener(conn: DuckDBPyConnection) -> DuckDBPyRelation:
            return self.to_duckdb(conn).select(*[f'"{col}"' for col in cols])
        return DatasetSource.from_duckdb_parser(opener, new_schema, batch_size)

    def map(self, new_schema: Schema, mapper: Callable[[Dataset], Dataset]) -> DatasetSourceFromFunction:
        """Apply a function to each Dataset returned by open().

        The provided new_schema must be correct. This is a low-level method.
        """
        return DatasetSourceFromFunction(new_schema, lambda conn: (mapper(dataset) for dataset in self.open(conn)))

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
    def from_table_name(table_name: DuckdbName | str, conn: DuckDBPyConnection, batch_size: int = default_duckdb_batch_size) -> DatasetSource:
        return DatasetSource.from_table(DuckdbTable.from_duckdb(table_name, conn), batch_size)

    @staticmethod
    def from_duckdb_parser(opener: Callable[[DuckDBPyConnection], DuckDBPyRelation],
                           conn_or_schema: DuckDBPyConnection | Schema,
                           batch_size: int = default_duckdb_batch_size) -> DatasetSource:
        """Read any data that duckdb can access by supplying an explicit query or method call on a Connection.

        Args:
             conn_or_schema: if the Schema is known, a DatasetSource is returned immediately.
                             Otherwise, a connection must be given, and the dataset will be opened once
                             (but not read fully) in order to find out its schema.
        """
        from agentune.analyze.core.duckdbio import DatasetSourceFromDuckdb, sniff_schema
        if isinstance(conn_or_schema, DuckDBPyConnection):
            return sniff_schema(opener, conn_or_schema, batch_size)
        else:
            return DatasetSourceFromDuckdb(conn_or_schema, opener, batch_size)

    @staticmethod
    def from_csv(path: Path | httpx.URL | str | StringIO, conn: DuckDBPyConnection,
                 header: bool | int | None = None, compression: str | None = None, sep: str | None = None,
                 delimiter: str | None = None, dtype: dict[str, str] | list[str] | None = None,
                 na_values: str | list[str] | None = None, skiprows: int | None = None,
                 quotechar: str | None = None, escapechar: str | None = None, encoding: str | None = None,
                 parallel: bool | None = None, date_format: str | None = None,
                 timestamp_format: str | None = None, sample_size: int | None = None,
                 all_varchar: bool | None = None, normalize_names: bool | None = None,
                 null_padding: bool | None = None, names: list[str] | None = None,
                 lineterminator: str | None = None, columns: dict[str, str] | None = None,
                 auto_type_candidates: list[str] | None = None, max_line_size: int | None = None,
                 ignore_errors: bool | None = None, store_rejects: bool | None = None,
                 rejects_table: str | None = None, rejects_scan: str | None = None,
                 rejects_limit: int | None = None, force_not_null: list[str] | None = None,
                 buffer_size: int | None = None, decimal: str | None = None,
                 allow_quoted_nulls: bool | None = None, filename: bool | str | None = None,
                 hive_partitioning: bool | None = None, union_by_name: bool | None = None,
                 hive_types: dict[str, str] | None = None, hive_types_autocast: bool | None = None,
                 batch_size: int = default_duckdb_batch_size) -> DatasetSource:
        """Read CSV data from a local path or remote URL, or from in-memory data represented with StringIO.

        CSV reading is implemented by duckdb and is configurable. The arguments are documented at
        https://duckdb.org/docs/stable/data/csv/overview.html, and match the signature of `duckdb.read_csv`.

        Note that duckdb supports reading any text stream (i.e. TextIOBase), but a DatasetSource can be read
        multiple times, so it can't use a stream that is only consumable once. If you need to read a text stream,
        please use duckdb directly.
        """
        from agentune.analyze.core.duckdbio import sniff_schema
        
        # Helper function to avoid parameter duplication 
        def make_csv_reader(connection: DuckDBPyConnection, file_path: Any) -> DuckDBPyRelation:
            return connection.read_csv(file_path,
                                      header=header, compression=compression, sep=sep,
                                      delimiter=delimiter, dtype=dtype, na_values=na_values,
                                      skiprows=skiprows, quotechar=quotechar, escapechar=escapechar,
                                      encoding=encoding, parallel=parallel, date_format=date_format,
                                      timestamp_format=timestamp_format, sample_size=sample_size,
                                      all_varchar=all_varchar, normalize_names=normalize_names,
                                      null_padding=null_padding, names=names,
                                      lineterminator=lineterminator, columns=columns,
                                      auto_type_candidates=auto_type_candidates, max_line_size=max_line_size,
                                      ignore_errors=ignore_errors, store_rejects=store_rejects,
                                      rejects_table=rejects_table, rejects_scan=rejects_scan,
                                      rejects_limit=rejects_limit, force_not_null=force_not_null,
                                      buffer_size=buffer_size, decimal=decimal,
                                      allow_quoted_nulls=allow_quoted_nulls, filename=filename,
                                      hive_partitioning=hive_partitioning, union_by_name=union_by_name,
                                      hive_types=hive_types, hive_types_autocast=hive_types_autocast)
        
        if isinstance(path, Path | httpx.URL):
            path = str(path)
        elif isinstance(path, StringIO):
            # For StringIO, we need to capture the content and create new instances each time
            content = path.getvalue()
            return sniff_schema(lambda conn: make_csv_reader(conn, StringIO(content)),
                                conn, batch_size=batch_size)
        
        return sniff_schema(lambda conn: make_csv_reader(conn, path),
                            conn, batch_size=batch_size)

    @staticmethod
    def from_parquet(path: Path | httpx.URL | str, conn: DuckDBPyConnection,
                     binary_as_string: bool = False, file_row_number: bool = False, filename: bool = False,
                     hive_partitioning: bool = False, union_by_name: bool = False, compression: str | None = None,
                     batch_size: int = default_duckdb_batch_size) -> DatasetSource:
        """Read Parquet, from a local path or remote URL or from in-memory data or a stream.

        Parquet reading is implemented by duckdb and is configurable. The arguments are documented at
        https://duckdb.org/docs/stable/data/parquet/overview.html, and match the signature of `duckdb.read_parquet`.
        """
        from agentune.analyze.core.duckdbio import sniff_schema
        if isinstance(path, Path | httpx.URL):
            path = str(path)
        return sniff_schema(lambda conn: conn.read_parquet(path,
                                                           binary_as_string=binary_as_string, file_row_number=file_row_number,
                                                           filename=filename, hive_partitioning=hive_partitioning,
                                                           union_by_name=union_by_name, compression=compression),
                            conn, batch_size=batch_size)

    @staticmethod
    def from_json(path: Path | httpx.URL | str | StringIO, conn: DuckDBPyConnection,
                  columns: dict[str, str] | None = None, sample_size: int | None = None,
                  maximum_depth: int | None = None, records: str | None = None, format: str | None = None,
                  date_format: str | None = None, timestamp_format: str | None = None,
                  compression: str | None = None, maximum_object_size: int | None = None,
                  ignore_errors: bool | None = None, convert_strings_to_integers: bool | None = None,
                  field_appearance_threshold: float | None = None, map_inference_threshold: int | None = None,
                  maximum_sample_files: int | None = None, filename: bool | str | None = None,
                  hive_partitioning: bool | None = None, union_by_name: bool | None = None,
                  hive_types: dict[str, str] | None = None, hive_types_autocast: bool | None = None,
                  batch_size: int = default_duckdb_batch_size) -> DatasetSource:
        """Read ndjson (newline-delimited json records), from a local path or remote URL or from in-memory data or a stream.

        Json reading is implemented by duckdb and is configurable. The arguments are documented at
        https://duckdb.org/docs/stable/data/json/overview.html, and match the signature of `duckdb.read_json`.

        Note that duckdb supports reading any text stream (i.e. TextIOBase), but a DatasetSource can be read
        multiple times, so it can't use a stream that is only consumable once. If you need to read a text stream,
        please use duckdb directly.
        """
        from agentune.analyze.core.duckdbio import sniff_schema
        
        # Helper function to avoid parameter duplication 
        def make_json_reader(connection: DuckDBPyConnection, file_path: Any) -> DuckDBPyRelation:
            return connection.read_json(file_path,
                                       columns=columns, sample_size=sample_size,
                                       maximum_depth=maximum_depth, records=records, format=format,
                                       date_format=date_format, timestamp_format=timestamp_format,
                                       compression=compression, maximum_object_size=maximum_object_size,
                                       ignore_errors=ignore_errors, convert_strings_to_integers=convert_strings_to_integers,
                                       field_appearance_threshold=field_appearance_threshold,
                                       map_inference_threshold=map_inference_threshold,
                                       maximum_sample_files=maximum_sample_files, filename=filename,
                                       hive_partitioning=hive_partitioning, union_by_name=union_by_name,
                                       hive_types=hive_types, hive_types_autocast=hive_types_autocast)
        
        if isinstance(path, Path | httpx.URL):
            path = str(path)
        elif isinstance(path, StringIO):
            # For StringIO, we need to capture the content and create new instances each time
            content = path.getvalue()
            return sniff_schema(lambda conn: make_json_reader(conn, StringIO(content)),
                                conn, batch_size=batch_size)
        
        return sniff_schema(lambda conn: make_json_reader(conn, path),
                            conn, batch_size=batch_size)

@define
class OpaqueDatasetSource(DatasetSource):
    """Intermediate implementation class"""
    @override
    def to_duckdb(self, conn: DuckDBPyConnection) -> DuckDBPyRelation:
        return restore_relation_types(conn.from_arrow(self.to_arrow_reader(conn)), self.schema)

    @override
    def to_dataset(self, conn: DuckDBPyConnection) -> Dataset:
        iterator = self.open(conn)
        df = next(iterator).data
        for more in iterator:
            df = df.vstack(more.data, in_place=True)
        return Dataset(self.schema, df)

    @override
    def copy_to_thread(self) -> OpaqueDatasetSource:
        return self.map(self.schema, lambda dataset: dataset.copy_to_thread())


@frozen
class DatasetSourceFromIterable(OpaqueDatasetSource):
    schema: Schema
    iterable: Iterable[Dataset]

    @override
    def open(self, conn: DuckDBPyConnection) -> Iterator[Dataset]:
        return iter(self.iterable)


@frozen
class DatasetSourceFromDataset(DatasetSource):
    dataset: Dataset

    @property
    @override
    def schema(self) -> Schema:
        return self.dataset.schema

    def cheap_size(self, _conn: DuckDBPyConnection) -> int | None:
        return self.dataset.height

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


@frozen
class DatasetSourceFromFunction(OpaqueDatasetSource):
    schema: Schema
    function: Callable[[DuckDBPyConnection], Iterator[Dataset]]

    @override
    def open(self, conn: DuckDBPyConnection) -> Iterator[Dataset]:
        return self.function(conn)


    
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
    def into_duckdb_table(table_name: DuckdbName,
                          create_table: bool = True,
                          or_replace: bool = True, delete_contents: bool = True) -> "DuckdbTableSink":
        """See DuckdbDatasetSink for the arguments."""
        # Local import to avoid circle
        from agentune.analyze.core.duckdbio import DuckdbTableSink
        return DuckdbTableSink(table_name, create_table, or_replace, delete_contents)

    @staticmethod
    def into_unqualified_duckdb_table(table_name: str, conn: DuckDBPyConnection,
                                      create_table: bool = True,
                                      or_replace: bool = True, delete_contents: bool = True) -> "DuckdbTableSink":
        """See DuckdbDatasetSink for the arguments."""
        return DatasetSink.into_duckdb_table(DuckdbName.qualify(table_name, conn), create_table, or_replace, delete_contents)

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
