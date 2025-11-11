from __future__ import annotations

import datetime
from collections.abc import Sequence
from typing import Literal, override

import polars as pl
from attrs import field, frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.core.database import (
    ArtIndex,
    DuckdbIndex,
    DuckdbName,
    DuckdbTable,
)
from agentune.analyze.core.dataset import Dataset, duckdb_to_dataset
from agentune.analyze.core.schema import Field
from agentune.analyze.feature.dedup_names import deduplicate_strings
from agentune.analyze.join.base import (
    JoinStrategy,
)


@frozen
class TimeWindow:
    start: datetime.datetime
    end: datetime.datetime
    include_start: bool
    include_end: bool
    sample_maxsize: int | None = None

@frozen
class TimeSeries:
    """A single (unkeyed, potentially sliced) TimeSeries is represented as a dataframe with a datetime column and one or more value columns."""

    dataset: Dataset
    date_col_name: str
    date_col: Field = field(init=False)
    value_cols: Sequence[Field] = field(init=False)

    @date_col.default
    def _date_col_default(self) -> Field:
        return self.dataset.schema[self.date_col_name]
    
    @value_cols.default
    def _value_cols_default(self) -> list[Field]:
        return [f for f in self.dataset.schema.cols if f.name != self.date_col_name]
    
    def slice(self, window: TimeWindow) -> TimeSeries:
        """WARNING: the downsampling logic does not produce the same result as slicing inside duckdb.
        
        This is mostly useful for tests, not for implementing features, since unsliced and unsampled time series
        shouldn't normally appear as values in memory.
        """
        closed: Literal['left', 'right', 'both', 'none'] = \
            'both' if window.include_start and window.include_end else \
            'left' if window.include_start else \
            'right' if window.include_end else \
            'none'
        new_df: pl.DataFrame = self.dataset.data.filter(
            pl.col(self.date_col_name).is_between(window.start, window.end, closed=closed)
        )
        if window.sample_maxsize is not None:
            new_df = new_df.sample(n=window.sample_maxsize, seed=42)
        return TimeSeries(Dataset(self.dataset.schema, new_df), self.date_col_name)


@frozen
class KtsJoinStrategy[K](JoinStrategy):
    """A time-series join strategy. KTS stands for keyed time series."""

    name: str
    table: DuckdbTable
    key_col: Field
    date_col: Field # Should be of type timestamp
    value_cols: tuple[Field, ...] # can be used to restrict the available value columns from what's in the table

    @staticmethod
    def on_table(name: str, table: DuckdbTable, key_col: str, date_col: str, *value_cols: str) -> KtsJoinStrategy[K]:
        relevant_table = DuckdbTable(table.name, table.schema.select(*(key_col, date_col, *value_cols)))
        return KtsJoinStrategy[K](
            name, relevant_table,
            table.schema[key_col],
            table.schema[date_col],
            tuple(table.schema[col] for col in value_cols)
        )

    @property
    @override
    def index(self) -> DuckdbIndex:
        return ArtIndex(
            name=DuckdbName(f'art_by_{self.key_col.name}', self.table.name.database, self.table.name.schema),
            cols=(self.key_col.name, )
        )

    def get(self, conn: DuckDBPyConnection, key: K, window: TimeWindow, value_cols: Sequence[str]) -> TimeSeries | None: 
        start_op = '>=' if window.include_start else '>'
        end_op = '<=' if window.include_end else '<'
        sample_clause = f'USING SAMPLE {window.sample_maxsize} (reservoir, 42)' if window.sample_maxsize else ''
        
        # Sampling is not deterministic when multithreaded, even when a random seed is provided.
        # And we can only set threads globally (per database), not locally (per connection), which is too expensive
        # and would disrupt any currently running queries.
        # We need to do better. #189
        try:
            if window.sample_maxsize:
                #conn.execute('set threads = 1')
                pass

            # Get unique name for key_exists column that doesn't shadow any other column we need
            key_exists_col = deduplicate_strings(['key_exists'], [*value_cols, self.key_col.name, self.date_col.name])[0]

            # Join key_exists to get a single row of nulls if the key is not found
            # NOTE the USING SAMPLE clause applies to the table, after joins but before any WHERE filtering,
            #  so we need to use a join with a subquery instead of a simple filter on key and dates
            relation = conn.sql(f'''
                WITH 
                    key_exists AS (
                        SELECT exists(
                            SELECT 1 FROM {self.table.name} WHERE "{self.key_col.name}" = $key
                        ) AS {key_exists_col}
                    ),
                    main_table as (
                        SELECT "{self.date_col.name}",
                                {", ".join(f'"{col}"' for col in value_cols)}
                        FROM {self.table.name}
                        WHERE "{self.key_col.name}" = $key
                            AND "{self.date_col.name}" {start_op} $start
                            AND "{self.date_col.name}" {end_op} $end
                        ORDER BY "{self.date_col.name}"
                    )
                SELECT key_exists.{key_exists_col}, main_table.*
                FROM key_exists
                LEFT JOIN main_table on 1
                {sample_clause}
            ''', params={'key': key, 'start': window.start, 'end': window.end})
            
            dataset = duckdb_to_dataset(relation)
            dataset_without_key_exists = dataset.drop(key_exists_col)

            # Handle special cases
            if dataset.data.height == 1:
                key_found = dataset.data[key_exists_col].any()
                if key_found:
                    if dataset_without_key_exists.data[self.date_col.name].is_null().all():
                        # Key found but no data in time range; return empty time series
                        return TimeSeries(dataset_without_key_exists.empty(), self.date_col.name)
                else:
                    # key not found
                    return None        
           
            return TimeSeries(dataset_without_key_exists, self.date_col.name)
        finally:
            if window.sample_maxsize:
                #conn.execute('reset threads')
                pass

    # I wanted to implemented a get_batch, taking `keys: Sequence[K]` and returning `Sequence[Option[TimeSeries]]`,
    # but I don't know how to implement the downsampling in a way that would be more efficient than calling it once per key.

