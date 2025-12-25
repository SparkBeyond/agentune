import enum
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol, override

import polars as pl
from attrs import define, frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.feature.base import (
    BoolFeature,
    CategoricalFeature,
    FloatFeature,
    IntFeature,
    SqlQueryFeature,
    SyncFeature,
)
from agentune.analyze.feature.validate.base import FeatureValidationCode, FeatureValidationError
from agentune.analyze.join.base import JoinStrategy
from agentune.core import types
from agentune.core.database import DuckdbTable
from agentune.core.dataset import Dataset, DatasetSourceFromDataset, duckdb_to_polars
from agentune.core.schema import Field, Schema


def _register_input_table_with_index(conn: DuckDBPyConnection, dataset: Dataset,
                                     primary_table_name: str, index_column_name: str) -> None:
    input_with_index_data = dataset.data.with_row_index(index_column_name)
    input_with_index_schema = Schema((Field(index_column_name, types.uint32), *dataset.schema.cols))
    input_with_index = Dataset(input_with_index_schema, input_with_index_data)
    # Go through DatasetSourceFromDataset to make the registered relation have the right duckdb schema
    input_relation = DatasetSourceFromDataset(input_with_index).to_duckdb(conn)
    conn.register(primary_table_name, input_relation)

class ComputeValidationCode(FeatureValidationCode):
    width = enum.auto()
    height = enum.auto()
    dtype = enum.auto()

@define
class SqlBackedFeature[T](SqlQueryFeature, SyncFeature[T]):
    """A feature implemented as a single SQL query.

    The query can address the main table under the name self.primary_table_name.
    It must order it by self.index_column_name.
    These names MAY require quoting in the query.

    The index_column_name MAY NOT shadow a column that appears in the 'real' primary input table, even if the query
    doesn't use that column. This restriction may be lifted in the future.

    Note that `primary_table` is a string, not a DuckdbName, because it is register()ed with the connection
    and only names in the current catalog and schema can be registered. That means the query can't assume
    that the original name that we are shadowing is from a different catalog or schema.

    A valid query must (this is not validated here):
    - Select FROM the ‘self.primary_table’ table
        - A query can be valid without strictly doing this, for example by adding a projection on top:
          SELECT a + b from (SELECT * from primary_table ... ORDER BY primary_table.rowid)
    - Not access the `self.primary_table` in any other way
    - Produce a result set with exactly one column, of a type valid for the current feature type:
        - For int features, int32 or any smaller int or uint type
        - For float features, float64 or float32
        - For bool features, bool
        - For categorical features, enum (of a type matching the categories list) or str
    - Return a result set with exactly one row per input primary table row, in a matching order
        - Sort the query by `{self.primary_table}.{self.index_column_name}` to achieve this.
    - Be deterministic (in particular, wrt. random sampling, and the native order of secondary tables)
    """
    sql_query: str

    primary_table_name: str
    index_column_name: str

    name: str
    description: str

    params: Schema
    secondary_tables: tuple[DuckdbTable, ...]
    join_strategies: tuple[JoinStrategy, ...]

    @override
    def compute_batch(self, input: Dataset, conn: DuckDBPyConnection) -> pl.Series:
        # Separate cursor to register the main table
        with conn.cursor() as cursor:
            _register_input_table_with_index(cursor, input, self.primary_table_name, self.index_column_name)
            relation = cursor.sql(self.sql_query)
            result = duckdb_to_polars(relation)
            if result.width != 1:
                raise FeatureValidationError(ComputeValidationCode.width,
                                             f'SQL query returned {result.width} columns instead of one')
            result_series = result.to_series(0)
            if result_series.len() != input.data.height:
                raise FeatureValidationError(ComputeValidationCode.height,
                                             f'SQL query returned wrong number of rows: {result_series.len()} outputs for {input.data.height} inputs')

            # Polars Series.cast() with these args will raise an error if the cast is invalid or loses numerical precision,
            # but only as far as the actual values go; if the result has a formal type of float64 but all the values
            # can be represented exactly as an int32, then it is still a valid int feature.
            # This is not a perfect defense: it will also lose precision when casting ints to floats (without raising an error),
            # and it is willing to cast anything to a string or a number to a boolean.
            # It only helps in the specific case of casting a float or a bigger int to a smaller int and losing precision.
            try:
                return result_series.cast(self.raw_dtype.polars_type, strict=True, wrap_numerical=False).rename(self.name)
            except pl.exceptions.InvalidOperationError as e:
                raise FeatureValidationError(ComputeValidationCode.dtype,
                                             f'SQL query returned type {relation.dtypes[0]} which cannot be cast to {self.dtype.duckdb_type}: {e}') from e


@frozen
class IntSqlBackedFeature(IntFeature, SqlBackedFeature):
    pass


@frozen
class FloatSqlBackedFeature(FloatFeature, SqlBackedFeature):
    pass


@frozen
class BoolSqlBackedFeature(BoolFeature, SqlBackedFeature):
    pass


@frozen
class CategoricalSqlBackedFeature(CategoricalFeature, SqlBackedFeature):
    pass



class SqlFeatureCorrector(ABC):
    """A callback that tries to fix a feature given a validation error, or gives up."""

    @abstractmethod
    async def correct(self, sql_query: str, error: FeatureValidationError) -> str | None:
        """Return a new SQL query to try, or None to give up."""
        ...


class FeatureFromQueryCtor(Protocol):
    """The signature of the feature_from_query_xxx functions defined in this package, allowing them to be passed as arguments
    with simpler type signatures.

    The extra default_for_xxx parameters that some of these functions have are not included here.
    """

    def __call__(self,
                 conn: DuckDBPyConnection,
                 sql_query: str,
                 params: Schema, secondary_tables: Sequence[DuckdbTable],

                 primary_table_name: str = 'primary_table',
                 index_column_name: str = 'rowid',

                 name: str | None = None,
                 description: str = '',
                 technical_description: str | None = None) -> SqlBackedFeature: ...

