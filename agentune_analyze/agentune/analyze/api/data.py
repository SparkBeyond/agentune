from __future__ import annotations

import asyncio
from collections.abc import Callable
from io import StringIO
from pathlib import Path

import httpx
import polars as pl
from attrs import frozen
from duckdb import DuckDBPyConnection, DuckDBPyRelation

from agentune.analyze.api.base import RunContext
from agentune.analyze.core.database import DuckdbName, DuckdbTable
from agentune.analyze.core.dataset import (
    Dataset,
    DatasetSink,
    DatasetSource,
    IfTargetExists,
    ReadCsvParams,
    ReadNdjsonParams,
    ReadParquetParams,
    WriteCsvParams,
    WriteParquetParams,
)
from agentune.analyze.core.duckdbio import DuckdbTableSource
from agentune.analyze.core.schema import Schema
from agentune.analyze.join.conversation import ConversationJoinStrategy
from agentune.analyze.join.lookup import LookupJoinStrategy
from agentune.analyze.join.timeseries import KtsJoinStrategy
from agentune.analyze.run.ingest import sampling
from agentune.analyze.run.ingest.sampling import SplitDuckdbTable


@frozen
class BoundData:
    """Methods for reading and writing data bound to a RunContext."""
    run_context: RunContext

    def from_df(self, data: pl.DataFrame | Dataset | DatasetSource) -> BoundDatasetSource:
        """Link an existing data source to this context to use the methods of this API with it.

        This does not perform any IO or have side effects.
        """
        match data:
            case pl.DataFrame():
                return BoundDatasetSource(self.run_context, DatasetSource.from_dataset(Dataset.from_polars(data)))
            case Dataset():
                return BoundDatasetSource(self.run_context, DatasetSource.from_dataset(data))
            case DatasetSource():
                return BoundDatasetSource(self.run_context, data)

    def from_sql(self, query: str | Callable[[DuckDBPyConnection], DuckDBPyRelation], params: object | None = None,
                 batch_size: int | None = None) -> BoundDatasetSource:
        """Define a data source that, when consumed, runs the given SQL query in the duckdb database.

        Calling this method does not run the query, but it does evaluate it enough to determine its output schema
        and will fail if the query is invalid or references nonexistent tables.

        For the semantics of `sql` and `params`, see https://duckdb.org/docs/stable/clients/python/relational_api#sql
        and https://duckdb.org/docs/stable/clients/python/dbapi#prepared-statements.

        Args:
            query: if a string, it will have the effect of calling DuckdbPyConnection.sql(query, params).
                   if a callable, it will be called with the duckdb connection as the only argument;
                   the params object must be None.
        """
        match query:
            case str() as s:
                opener = lambda conn: conn.sql(s, params=params) # noqa: E731
            case _:
                if params is not None:
                    raise ValueError('params must be None when query is a callable')
                opener = query
        batch_size = batch_size or self.run_context.defaults.duckdb_batch_size
        with self.run_context._ddb_manager.cursor() as conn:
            return BoundDatasetSource(self.run_context, DatasetSource.from_duckdb_parser(opener, conn, batch_size))

    def from_csv(self, path: Path | httpx.URL | str | StringIO,
                 read_csv_params: ReadCsvParams = ReadCsvParams(),
                 batch_size: int | None = None) -> BoundDatasetSource:
        """Define a data source that, when consumed, reads a CSV file, or multiple files
        depending on read_csv_params.

        Calling this method does not read the whole file, but it reads enough of it to determine its output schema.
        """
        batch_size = batch_size or self.run_context.defaults.duckdb_batch_size
        with self.run_context._ddb_manager.cursor() as conn:
            dataset_source = DatasetSource.from_csv(path, conn, read_csv_params, batch_size)
            return BoundDatasetSource(self.run_context, dataset_source)

    def from_parquet(self, path: Path | httpx.URL | str,
                     read_parquet_params: ReadParquetParams = ReadParquetParams(),
                     batch_size: int | None = None) -> BoundDatasetSource:
        """Define a data source that, when consumed, reads a Parquet file, or multiple files
        depending on read_parquet_params.

        Calling this method does not read the whole file, but it reads enough of it to determine its output schema.
        """
        batch_size = batch_size or self.run_context.defaults.duckdb_batch_size
        with self.run_context._ddb_manager.cursor() as conn:
            dataset_source = DatasetSource.from_parquet(path, conn, read_parquet_params, batch_size)
            return BoundDatasetSource(self.run_context, dataset_source)

    def from_ndjson(self, path: Path | httpx.URL | str | StringIO,
                    read_ndjson_params: ReadNdjsonParams = ReadNdjsonParams(),
                    batch_size: int | None = None) -> BoundDatasetSource:
        """Define a data source that, when consumed, reads an NDJson (newline-delimited JSON) file.

        Calling this method does not read the whole file, but it reads enough of it to determine its output schema.
        """
        batch_size = batch_size or self.run_context.defaults.duckdb_batch_size
        with self.run_context._ddb_manager.cursor() as conn:
            dataset_source = DatasetSource.from_ndjson(path, conn, read_ndjson_params, batch_size)
            return BoundDatasetSource(self.run_context, dataset_source)

    def to_table(self, table: str | DuckdbName | DuckdbTable | BoundTable,
                 create_if_not_exists: bool = True,
                 if_exists: IfTargetExists = IfTargetExists.REPLACE) -> BoundDatasetSink:
        """Define a data sink that will write data to the named table, possibly creating or replacing it.

        Calling this method has no effects until the sink is used with e.g. BoundDatasetSource.copy_to.

        Args:
            table: a way of identifying a table.
            create_if_not_exists: if the table does not exist, creates it if True, or fails if False.
            if_exists: if the table already exists, whether to replace it, append to it, or fail.
                       Replacement and appending are both atomic.
        """
        match table:
            case str() as string_name:
                with self.run_context._ddb_manager.cursor() as conn:
                    name = DuckdbName.qualify(string_name, conn)
            case DuckdbName() as qual_name:
                name = qual_name
            case DuckdbTable() as table:
                name = table.name
            case BoundTable() as table:
                name = table.table.name
        return BoundDatasetSink(self.run_context, DatasetSink.into_duckdb_table(name, create_if_not_exists, if_exists))

    def to_csv(self, path: Path | str,
               params: WriteCsvParams = WriteCsvParams()) -> BoundDatasetSink:
        """Define a data sink that will write data to a CSV file.

        Calling this method has no effects until the sink is used with e.g. BoundDatasetSource.copy_to.
        """
        return BoundDatasetSink(self.run_context, DatasetSink.into_csv(path, params))

    def to_parquet(self, path: Path | str,
                   params: WriteParquetParams = WriteParquetParams()) -> BoundDatasetSink:
        """Define a data sink that, when used, will write data to a Parquet file or files.

        Calling this method has no effects until the sink is used with e.g. BoundDatasetSource.copy_to.
        """
        return BoundDatasetSink(self.run_context, DatasetSink.into_parquet(path, params))


@frozen
class BoundDatasetSource:
    """A DatasetSource bound to a RunContext instance.

    A DatasetSource can be read multiple times and has a known schema.
    """
    run_context: RunContext
    dataset_source: DatasetSource

    @property
    def schema(self) -> Schema:
        """The column schema of this dataset source."""
        return self.dataset_source.schema

    @property
    def cheap_size(self) -> int | None:
        """Return the size if it can be checked cheaply, or None otherwise.

        May perform a small amount of IO, but will not read the whole input to count the rows.
        """
        with self.run_context._ddb_manager.cursor() as conn:
            return self.dataset_source.cheap_size(conn)

    async def size(self) -> int:
        """Returns the size of the dataset source.

        May read the entire source if necessary to determine the size (e.g., if it's a CSV file).
        See also `self.cheap_size`.
        """
        s = self.cheap_size
        if s is not None:
            return s
        def blocking_size(conn: DuckDBPyConnection, dataset_source: DatasetSource) -> int:
            return len(dataset_source.to_duckdb(conn))
        with self.run_context._ddb_manager.cursor() as conn:
            return await asyncio.to_thread(blocking_size, conn, self.dataset_source.copy_to_thread())

    async def load(self) -> Dataset:
        """Read the entire source into memory as a Dataset.

        A Dataset combines a Schema with a Polars DataFrame containing the actual data.
        This method will read all data from the source, which can take a long time and can run out of memory.
        """
        with self.run_context._ddb_manager.cursor() as conn:
            return await asyncio.to_thread(self.dataset_source.copy_to_thread().to_dataset, conn.cursor())

    def sample_as_string(self, sample_size: int = 10) -> str:
        """Return a string representation of the first `sample_size` rows, formatted as a table with column names and types."""
        with self.run_context._ddb_manager.cursor() as conn:
            relation = self.dataset_source.to_duckdb(conn).limit(sample_size)
            return str(relation)

    def print_sample(self, sample_size: int = 10) -> None:
        """print() a string representation of the first `sample_size` rows, formatted as a table with column names and types."""
        print(self.sample_as_string(sample_size)) # noqa: T201

    async def copy_to(self, target: BoundDatasetSink | DatasetSink) -> BoundDatasetSink:
        """Copy the contents of this dataset source to the given location."""
        if isinstance(target, BoundDatasetSink):
            if target.run_context is not self.run_context:
                raise ValueError('Cannot copy to a different RunContext')
            target = target.dataset_sink
        with self.run_context._ddb_manager.cursor() as conn:
            await asyncio.to_thread(target.write, self.dataset_source.copy_to_thread(), conn)
        return BoundDatasetSink(self.run_context, target)

    async def copy_to_table(self, table: str | DuckdbName | DuckdbTable,
                            create_if_not_exists: bool = True,
                            if_exists: IfTargetExists = IfTargetExists.REPLACE,
                            batch_size: int | None = None) -> BoundTable:
        """Copy the contents of this dataset source to the given database table.

        Equivalent to `copy_to(ctx.data.to_table(...))` and also returns the BoundTable for convenient chaining.

        The copy_to method cannot return a ContextDatasetSource corresponding to its target
        because in some cases we don't know enough to construct the corresponding source, e.g. with a manually constructed
        DatasetSinkToDuckdb.
        """
        batch_size = batch_size or self.run_context.defaults.duckdb_batch_size
        await self.copy_to(self.run_context.data.to_table(table, create_if_not_exists, if_exists))
        with self.run_context._ddb_manager.cursor() as conn:
            match table:
                case str():
                    name = DuckdbName.qualify(table, conn)
                case DuckdbName():
                    name = table
                case DuckdbTable():
                    name = table.name
            table = DuckdbTable.from_duckdb(name, conn)
            return BoundTable(self.run_context, table, DuckdbTableSource(table, batch_size))

@frozen
class BoundDatasetSink:
    """A DatasetSink bound to a RunContext instance.

    A DatasetSink describes a location and format where data can be written.
    It does not indicate whether that location exists or contains data; any error will be raised when it is used,
    not when it is created.
    """
    run_context: RunContext
    dataset_sink: DatasetSink


@frozen
class BoundTable(BoundDatasetSource):
    """A database table description bound to a RunContext instance and usable as a dataset source."""
    table: DuckdbTable
    dataset_source: DuckdbTableSource

    @property
    def name(self) -> DuckdbName:
        return self.table.name

    @property
    def join_strategy(self) -> BoundJoinStrategy:
        """Methods for defining join strategies on this table."""
        return BoundJoinStrategy(self.run_context, self.table)

    async def split(self, if_not_exists: bool = True,
                    train_fraction: float = 0.8,
                    feature_search_size: int = sampling.default_feature_search_size,
                    feature_eval_size: int = sampling.default_feature_eval_size,
                    is_train_col_name: str = '_is_train',
                    is_feature_search_col_name: str = '_is_feature_search',
                    is_feature_eval_col_name: str = '_is_feature_eval') -> BoundSplitTable:
        """Split the table between train, test, feature search and feature eval subsets, referred to as 'splits'.

        These are the four datasets needed to run the analyzer. See `ops.analyze`, `AnalyzeInputData`
        and `AnalyzeRunner` for more details.

        Splitting is implemented by adding several columns to the table, indicating which split each row belongs to.

        First, train_fraction of the entire table is assigned as train; the rest is test.
        Then, feature_search_size and feature_eval_size rows are selected from the train split.

        The feature search and feature eval splits can overlap (partially or entirely), and they can be smaller than
        the specified number of rows if the train split is too small.
        The train split is inclusive of the feature search and feature eval splits.

        Pass train_fraction=1.0 to leave the test split empty; this is useful if you intend to provide a separate
        test dataset to `ops.run_feature_search` later.

        The splitting is deterministic; in a table of the same size, the same row indexes are always assigned to
        the same splits. The random seed cannot be altered.

        The proportional split between train and test is accurate to within 0.01%; the splits that specify a number
        of rows are fully accurate.

        Args:
            if_not_exists: if the split columns already exist, do nothing if True, or raise an error if False.
            train_fraction: fraction of entire dataset's rows to assign to train; the rest are assigned to test.
            feature_search_size: number of rows out of the train split to assign to the feature search split.
                                 If the train split is smaller than this, the feature search split will be smaller.
            feature_eval_size: number of rows out of the train split to assign to the feature eval split.
                               If the train split is smaller than this, the feature eval split will be smaller.
            is_train_col_name: name of the column that will be created to mark the train-split. Rows where the column
                               is false are assigned to the test split.
            is_feature_search_col_name: name of the column that will be created to mark the feature search split.
                                        Rows in the feature search split also belong to the train split.
            is_feature_eval_col_name: name of the column that will be created to mark the feature eval split.
                                      Rows in the feature eval split also belong to the train split.
        """
        all_split_cols_exist = is_train_col_name in self.schema.names and is_feature_eval_col_name in self.schema.names \
                               and is_feature_search_col_name in self.schema.names
        any_split_cols_exist = is_train_col_name in self.schema.names or is_feature_eval_col_name in self.schema.names \
                               or is_feature_search_col_name in self.schema.names

        if not if_not_exists and any_split_cols_exist:
            raise ValueError('Some of the split columns already exist in the table')
        elif if_not_exists and all_split_cols_exist:
            return BoundSplitTable(self.run_context, self.dataset_source,
                                   SplitDuckdbTable(self.table, is_train_col_name, is_feature_search_col_name,
                                                           is_feature_eval_col_name))
        else:
            with self.run_context._ddb_manager.cursor() as conn:
                split = await asyncio.to_thread(sampling.split_duckdb_table, conn,
                                                self.table.name, train_fraction, feature_search_size,
                                                feature_eval_size,
                                                is_train_col_name, is_feature_search_col_name,
                                                is_feature_eval_col_name)
                return BoundSplitTable(self.run_context, self.dataset_source, split)

@frozen
class BoundSplitTable(BoundDatasetSource):
    """A split table bound to a RunContext instance.

    A split table is a table with columns marking train/test and feature search + feature eval splits of its rows.
    Train contains feature search and feature eval. Feature search and feature eval can overlap.
    """
    dataset_source: DuckdbTableSource
    splits: SplitDuckdbTable

    @property
    def table(self) -> DuckdbTable:
        return self.splits.table

    @property
    def name(self) -> DuckdbName:
        return self.table.name

    @property
    def train(self) -> BoundDatasetSource:
        return BoundDatasetSource(self.run_context, self.splits.train())

    @property
    def test(self) -> BoundDatasetSource:
        return BoundDatasetSource(self.run_context, self.splits.test())

    @property
    def feature_search(self) -> BoundDatasetSource:
        return BoundDatasetSource(self.run_context, self.splits.feature_search())

    @property
    def feature_eval(self) -> BoundDatasetSource:
        return BoundDatasetSource(self.run_context, self.splits.feature_eval())

@frozen
class BoundJoinStrategy:
    """Methods for defining join strategies on a table, bound to a RunContext instance."""
    run_context: RunContext
    table: DuckdbTable

    def lookup(self, name: str, key_col: str, *value_cols: str) -> LookupJoinStrategy:
        """Define a lookup join strategy, equivalent to an inner join.

        The key column is expected to be unique.
        """
        return LookupJoinStrategy.on_table(name, self.table, key_col, *value_cols)

    def keyed_time_series(self, name: str, key_col: str, date_col: str, *value_cols: str) -> KtsJoinStrategy:
        """Define a keyed time series join strategy.

        The table should have a key, date, and one or more value columns.
        The strategy allows querying by a key and range of dates (defined using class `TimeWindow`),
        returning a list of date and per-date values from each value column.
        """
        return KtsJoinStrategy.on_table(name, self.table, key_col, date_col, *value_cols)

    def conversation(self, name: str, main_table_key_col: str, key_col: str, timestamp_col: str,
                     role_col: str, content_col: str) -> ConversationJoinStrategy:
        """Define a conversation join strategy.

        A conversations table should have one row per message in a conversation.
        Each row should have a conversation ID, timestamp, role (e.g. 'user' or 'assistant'), and content (the message).
        For a given key, the conversation can be retrieved, represented by class `Conversation`.

        Args:
            main_table_key_col: the column in the main table (not this table) to use to look up conversations.
                                Must have the same type as the key_col column.
            key_col:            the column in this table to use to look up conversations.
            timestamp_col:      a column in this table containing the timestamp of each message.
                                Must be of type date, time or datetime.
            role_col:           a column in this table containing the role (i.e. sender) of each message.
                                Must be of type string.
            content_col:        a column in this table containing the content of each message.
                                Must be of type string.
        """
        return ConversationJoinStrategy.on_table(name, self.table, main_table_key_col, key_col, timestamp_col, role_col, content_col)
