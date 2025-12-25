import enum
import math
import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast, override

import duckdb
import polars as pl
from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.feature.base import Feature
from agentune.analyze.feature.validate.base import (
    FeatureValidationCode,
    FeatureValidationError,
    FeatureValidator,
)
from agentune.core import database, types
from agentune.core.dataset import Dataset
from agentune.core.schema import Field, Schema
from agentune.core.util.duckdbutil import ConnectionWithInit


class LawAndOrderValidationCode(FeatureValidationCode):
    consistent = enum.auto()
    compute_error = enum.auto()
    constant = enum.auto()
    output_len = enum.auto()
    row_by_row = enum.auto()
    null_to_null = enum.auto()
    access = enum.auto()
    input_order = enum.auto()


@frozen
class LawAndOrderValidator(FeatureValidator):
    """Validates these rules for a feature:

    - Behavior is consistent over multiple calls
    - Behavior is consistent between batched call and row-by-row calls (tested for `rows_to_compute_individually` random rows)
    - Zero rows are returned for empty main input
    - No errors are raised
    - Feature does not always return null, or always NaN, or always the same value
    - Feature does not access input or secondary columns that it does not declare
    - Output is the same per row regarding of the (natural) ordering of the input table

    Does not check if the feature depends on the natural ordering of any of the secondary tables.
    Reordering the secondary tables without making a full copy of each or defining a query that reads and fully sorts each
    is an unsolved problem. And making a copy of each of them is too expensive in the general case. So we don't validate it
    for now.
    """

    rows_to_compute_individually: int = 10

    @override
    async def validate(self, feature: Feature, input: Dataset, conn: DuckDBPyConnection) -> None:
        with self._with_limited_schema(feature, conn) as cursor:
            input = input.select(*feature.params.names)

            series1 = await self._acompute_batch(feature, input, cursor)
            self._sanity_check_series_result(input, series1)

            series2 = await self._acompute_batch(feature, input, cursor)
            if not series1.equals(series2, check_dtypes=True, check_names=True):
                raise FeatureValidationError(LawAndOrderValidationCode.consistent,
                                             'Feature output is not consistent between multiple computations on the same input')

            await self._test_input_order(feature, input, conn, series1)
            await self._test_row_by_row(feature, input, cursor, series1)

    async def _acompute_batch(self, feature: Feature, input: Dataset, conn: DuckDBPyConnection) -> pl.Series:
        try:
            return await feature.acompute_batch(input, conn)
        except duckdb.BinderException as e:
            # We assume the feature's query passed validation when the feature was constructed,
            # and so this must be due to our restricting the available schema.
            raise FeatureValidationError(LawAndOrderValidationCode.access, f'Feature accessed a column or table that it did not declare: {e}') from e
        except Exception as e:
            raise FeatureValidationError(LawAndOrderValidationCode.compute_error, f'Feature computation raised an error: {e}') from e

    async def _acompute(self, feature: Feature, args: tuple[Any, ...], conn: DuckDBPyConnection) -> Any:
        try:
            return await feature.acompute(args, conn)
        except duckdb.BinderException as e:
            raise FeatureValidationError(LawAndOrderValidationCode.access, f'Feature accessed a column or table that it did not declare: {e}') from e
        except Exception as e:
            raise FeatureValidationError(LawAndOrderValidationCode.compute_error, f'Feature computation raised an error: {e}') from e

    @contextmanager
    def _with_limited_schema(self, feature: Feature, conn: DuckDBPyConnection) -> Iterator[DuckDBPyConnection]:
        """Create a temp schema and inside it views that mimic the original secondary tables,
        but include only the tables and columns that the feature declares it needs.

        Returns:
            A ConnectionWithInit bound to access the temp schema
        """
        orig_table_names = [cast(str, row[0]) for row in
                            conn.execute('select table_name from duckdb_tables() '
                                         'where database_name = current_database() and schema_name = current_schema()').fetchall()]
        with database.temp_schema(conn, 'restrict_schema') as temp_schema:
            for table in feature.secondary_tables:
                col_specs = [f'"{c.name}"' for c in table.schema.cols]
                conn.execute(f'create view {temp_schema}."{table.name.name}" as select {', '.join(col_specs)} from {table.name}')

            # When a query accesses an unqualified table name which doesn't exist in the current schema, duckdb searches all other schemas
            # in the current catalog. So we need to shadow the names of all other tables in the original schema.
            # A query can still get around this by accessing a qualified name. I think the only way to stop it would be to
            # give it a dedicated database connection which really and truly doesn't contain the irrelevant tables,
            # but this is rather expensive to implement.
            # It might be better to parse the json explain output and see what the query actually accesses.
            secondary_table_names = [table.name.name for table in feature.secondary_tables]
            for name in orig_table_names:
                if name not in secondary_table_names:
                    conn.execute(f'create view {temp_schema}."{name}" as select col0 as __nonesuch_col__ from (values(1)) limit 0')

            use = cast(DuckDBPyConnection, ConnectionWithInit.use(conn, temp_schema))
            with use.cursor() as cursor:
                yield cursor


    def _sanity_check_series_result(self, input: Dataset, series: pl.Series) -> None:
        if series.is_null().all():
            raise FeatureValidationError(LawAndOrderValidationCode.constant, 'Feature always returns null')
        if series.dtype.is_float() and series.is_nan().all():
            raise FeatureValidationError(LawAndOrderValidationCode.constant, 'Feature always returns NaN')
        if series.n_unique() == 1:
            raise FeatureValidationError(LawAndOrderValidationCode.constant, f'Feature always returns the same value: {series[0]}')
        if series.len() != len(input.data):
            raise FeatureValidationError(LawAndOrderValidationCode.output_len,
                                         f'Feature returned wrong number of values: {series.len()} outputs for {input.height} inputs')

    async def _test_row_by_row(self, feature: Feature, input: Dataset, conn: DuckDBPyConnection,
                               all_rows_result: pl.Series) -> None:
        row_indexes = set(random.Random(42).sample(range(input.data.height), min(self.rows_to_compute_individually, input.data.height)))
        for index in row_indexes:
            args_by_name = input.data.row(index, named=True)
            args = tuple(args_by_name[name] for name in feature.params.names)
            row_result = await self._acompute(feature, args, conn)
            if row_result != all_rows_result[index] and not \
                    (isinstance(row_result, float) and math.isnan(row_result) and math.isnan(all_rows_result[index])):
                raise FeatureValidationError(LawAndOrderValidationCode.row_by_row,
                                             f'Row-by-row computation differs from batch computation: for inputs {args}, '
                                            f'row-by-row result was {row_result} and batch result was {all_rows_result[index]}')

    async def _test_input_order(self, feature: Feature, input: Dataset, conn: DuckDBPyConnection,
                                all_rows_result: pl.Series) -> None:
        index_col_name = '__index__'
        shuffled_input_with_index = Dataset(
            Schema((Field(index_col_name, types.uint32), *input.schema.cols)),
            input.data.with_row_index(index_col_name).sample(fraction=1.0, shuffle=True, seed=42)
        )
        shuffled_result = await feature.acompute_batch_safe(shuffled_input_with_index.drop(index_col_name), conn)
        order_mapping = shuffled_input_with_index.data[index_col_name].arg_sort()
        sorted_result = shuffled_result.gather(order_mapping)

        if not sorted_result.equals(all_rows_result, check_names=True, check_dtypes=True, null_equal=True):
            raise FeatureValidationError(LawAndOrderValidationCode.input_order,
                                         f'Feature {feature.name} behaves differently when the input is reordered.')
