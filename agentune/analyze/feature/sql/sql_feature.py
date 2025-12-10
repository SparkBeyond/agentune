from collections.abc import Sequence
from typing import override

import polars as pl
from _duckdb import DuckDBPyRelation
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
from agentune.analyze.join.base import JoinStrategy
from agentune.core import types
from agentune.core.database import DuckdbTable
from agentune.core.dataset import Dataset, DatasetSourceFromDataset, duckdb_to_polars
from agentune.core.schema import Field, Schema
from agentune.core.types import Dtype


def _register_input_table_with_index(conn: DuckDBPyConnection, dataset: Dataset,
                                     primary_table_name: str, index_column_name: str) -> None:
    input_with_index_data = dataset.data.with_row_index(index_column_name, dataset.data.width)
    input_with_index_schema = dataset.schema + Field(index_column_name, types.uint32)
    input_with_index = Dataset(input_with_index_schema, input_with_index_data)
    # Go through DatasetSourceFromDataset to make the registered relation have the right duckdb schema
    input_relation = DatasetSourceFromDataset(input_with_index).to_duckdb(conn)
    conn.register(primary_table_name, input_relation)


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
            result = duckdb_to_polars(cursor.sql(self.sql_query))
            if result.width != 1:
                raise ValueError(f'SQL query must return exactly one column but returned {result.width}')
            result_series = result.to_series(0)
            if result_series.len() != input.data.height:
                raise ValueError(f'SQL query returned {result_series.len()} rows for {input.data.height} input rows')

            # Polars Series.cast() with these args will raise an error if the cast is invalid or loses numerical precision,
            # but only as far as the actual values go; if the result has a formal type of float64 but all the values
            # can be represented exactly as an int32, then it is still a valid int feature.
            # This is not a perfect defense: it will also lose precision when casting ints to floats (without raising an error),
            # and it is willing to cast anything to a string or a number to a boolean.
            # It only helps in the specific case of casting a float or a bigger int to a smaller int and losing precision.
            return result_series.cast(self.dtype.polars_type, strict=True, wrap_numerical=False).rename(self.name)


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


def _relation_from_query(conn: DuckDBPyConnection,
                         sql_query: str,
                         params: Schema,
                         primary_table_name: str,
                         index_column_name: str,
                         secondary_tables: Sequence[DuckdbTable]) -> DuckDBPyRelation:
    if index_column_name in params.names:
        raise ValueError(f'Input data already has a column named {index_column_name}')
    if primary_table_name in [table.name.name for table in secondary_tables]:
        raise ValueError(f"Primary table name {primary_table_name} shadows secondary table's local name")

    dataset = Dataset(params,
                      pl.DataFrame({
                          field.name: pl.Series(field.name, [], field.dtype.polars_type) for field in params.cols
                      }))
    _register_input_table_with_index(conn, dataset, primary_table_name, index_column_name)
    relation = conn.sql(sql_query)

    if len(relation.types) != 1:
        raise ValueError(
            f'SQL query must return exactly one column but returned {len(relation.types)}: {relation.columns}')

    return relation


def feature_from_query(conn: DuckDBPyConnection,
                       sql_query: str,
                       params: Schema, secondary_tables: Sequence[DuckdbTable],

                       primary_table_name: str = 'primary_table',
                       index_column_name: str = 'rowid',

                       name: str | None = None,
                       description: str = '',
                       technical_description: str | None = None) -> SqlBackedFeature:
    """Create an SqlBackedFeature of the appropriate type, if possible, or raise an error.

    The feature type is autodetected from the query.

    The default_if_xxx parameters, which are feature-type-specific, are set to their default values (e.g. 0 for int);
    it would be inconvenient to pass them in a signature that doesn't know which ones to expect. You can change them
    by evolving the returned instance.

    Args:
         conn: a connection where the secondary_tables are available, just as when calling Feature.compute.
               This is required to parse and bind the SQL query. However, the query will not be executed,
               and the tables can be empty.
               It is possible to write a wrapper method that connects to a new in-memory database and creates empty
               secondary tables according to the given schemas, skipping the need for a preexisting connection.
         sql_query: a query (i.e. a single SELECT statement), obeying the requirements documented in class SqlBackedFeature.
                    This method validates some but not all of these requirements: namely, that the query must return a result
                    set with a single column of a valid type.

                    If the query returns a result of a dtype which isn't exactly one of the dtypes features should have,
                    a Feature will still be constructed according to the following rules:
                    - int < int32 or uint <= 16 -> int32
                    - float32 -> float64
                    - string -> enum containing only the category 'nonesuch'

                    The last case enables creating a CategoricalSqlBackedFeature whose categories list is wrong,
                    calling compute (not compute_safe) to collect some returned values, and evolving it to contain
                    a correct list of categories. Note that calling compute_safe on such a returned Feature will substitute
                    CategoricalFeature.other_category for all returned values until you evolve its list of categories.

                    'nonesuch' is used because CategoricalFeature.categories is not allowed to be empty or to contain
                    CategoricalFeature.other_category.

        params: expected schema of the primary input table, which will be available to the query under the name primary_table_name.
        secondary_tables: names and schemas of the secondary input tales.
        primary_table_name: the name used by the query to refer to the primary input table. This table is not expected
                            to exist in `conn` and it will not be used if it does exist.
        index_column_name: name of a synthetic column with row indexes which will be added to the primary table.
                           The query needs to order the results by this column. May not shadow the name of a preexisting
                           column.
        name: name of the created feature. If None, uses the name of the query's output column.
        description: populates Feature.description.
        technical_description: populates Feature.technical_description. If None, set to the query string.
    """
    with conn.cursor() as cursor:
        relation = _relation_from_query(cursor, sql_query, params, primary_table_name, index_column_name, secondary_tables)

        name = name or relation.columns[0]
        technical_description = technical_description or sql_query

        dtype = Dtype.from_duckdb(relation.types[0])
        if dtype == types.boolean:
            return BoolSqlBackedFeature(sql_query=sql_query,
                                        params=params, secondary_tables=tuple(secondary_tables), join_strategies=(),
                                        primary_table_name=primary_table_name, index_column_name=index_column_name,
                                        name=name, description=description, technical_description=technical_description,
                                        default_for_missing=False)
        elif dtype in [types.float32, types.float64]:
            return FloatSqlBackedFeature(sql_query=sql_query,
                                         params=params, secondary_tables=tuple(secondary_tables), join_strategies=(),
                                         primary_table_name=primary_table_name, index_column_name=index_column_name,
                                         name=name, description=description,
                                         technical_description=technical_description,
                                         default_for_missing=0.0, default_for_nan=0.0, default_for_infinity=0.0,
                                         default_for_neg_infinity=0.0)
        elif dtype in [types.int8, types.uint8, types.int16, types.uint16, types.int32]:
            return IntSqlBackedFeature(sql_query=sql_query,
                                       params=params, secondary_tables=tuple(secondary_tables), join_strategies=(),
                                       primary_table_name=primary_table_name, index_column_name=index_column_name,
                                       name=name, description=description,
                                       technical_description=technical_description,
                                       default_for_missing=0)
        elif dtype == types.string:
            return CategoricalSqlBackedFeature(sql_query=sql_query,
                                               params=params, secondary_tables=tuple(secondary_tables),
                                               join_strategies=(),
                                               primary_table_name=primary_table_name,
                                               index_column_name=index_column_name,
                                               name=name, description=description,
                                               technical_description=technical_description,
                                               default_for_missing=CategoricalFeature.other_category,
                                               categories=('nonesuch',))
        elif isinstance(dtype, types.EnumDtype):
            return CategoricalSqlBackedFeature(sql_query=sql_query,
                                               params=params, secondary_tables=tuple(secondary_tables),
                                               join_strategies=(),
                                               primary_table_name=primary_table_name,
                                               index_column_name=index_column_name,
                                               name=name, description=description,
                                               technical_description=technical_description,
                                               default_for_missing=CategoricalFeature.other_category,
                                               categories=dtype.values)
        else:
            raise ValueError(f'SQL query returned unsupported type {dtype}')


def int_feature_from_query(conn: DuckDBPyConnection,
                           sql_query: str,
                           params: Schema, secondary_tables: Sequence[DuckdbTable],

                           primary_table_name: str = 'primary_table',
                           index_column_name: str = 'rowid',

                           name: str | None = None,
                           description: str = '',
                           technical_description: str | None = None,

                           default_for_missing: int = 0) -> IntSqlBackedFeature:
    """As feature_from_query, but always creates an Int feature and raises an error if the query returns a different type."""
    with conn.cursor() as cursor:
        relation = _relation_from_query(cursor, sql_query, params, primary_table_name, index_column_name, secondary_tables)

        name = name or relation.columns[0]
        technical_description = technical_description or sql_query

        dtype = Dtype.from_duckdb(relation.types[0])
        if dtype in [types.int8, types.uint8, types.int16, types.uint16, types.int32]:
            return IntSqlBackedFeature(sql_query=sql_query,
                                       params=params, secondary_tables=tuple(secondary_tables), join_strategies=(),
                                       primary_table_name=primary_table_name, index_column_name=index_column_name,
                                       name=name, description=description,
                                       technical_description=technical_description,
                                       default_for_missing=default_for_missing)
        else:
            raise ValueError(f'SQL query returned type {dtype} and not an integer type')


def bool_feature_from_query(conn: DuckDBPyConnection,
                            sql_query: str,
                            params: Schema, secondary_tables: Sequence[DuckdbTable],

                            primary_table_name: str = 'primary_table',
                            index_column_name: str = 'rowid',

                            name: str | None = None,
                            description: str = '',
                            technical_description: str | None = None,

                            default_for_missing: bool = False) -> BoolSqlBackedFeature:
    """As feature_from_query, but always creates a boolean feature and raises an error if the query returns a different type."""
    with conn.cursor() as cursor:
        relation = _relation_from_query(cursor, sql_query, params, primary_table_name, index_column_name, secondary_tables)

        name = name or relation.columns[0]
        technical_description = technical_description or sql_query

        dtype = Dtype.from_duckdb(relation.types[0])
        if dtype == types.boolean:
            return BoolSqlBackedFeature(sql_query=sql_query,
                                        params=params, secondary_tables=tuple(secondary_tables), join_strategies=(),
                                        primary_table_name=primary_table_name, index_column_name=index_column_name,
                                        name=name, description=description, technical_description=technical_description,
                                        default_for_missing=default_for_missing)
        else:
            raise ValueError(f'SQL query returned type {dtype} and not boolean')


def float_feature_from_query(conn: DuckDBPyConnection,
                             sql_query: str,
                             params: Schema, secondary_tables: Sequence[DuckdbTable],

                             primary_table_name: str = 'primary_table',
                             index_column_name: str = 'rowid',

                             name: str | None = None,
                             description: str = '',
                             technical_description: str | None = None,

                             default_for_missing: float = 0.0,
                             default_for_nan: float = 0.0,
                             default_for_infinity: float = 0.0,
                             default_for_neg_infinity: float = 0.0) -> FloatSqlBackedFeature:
    """As feature_from_query, but always creates a float feature and raises an error if the query returns a different type."""
    with conn.cursor() as cursor:
        relation = _relation_from_query(cursor, sql_query, params, primary_table_name, index_column_name, secondary_tables)

        name = name or relation.columns[0]
        technical_description = technical_description or sql_query

        dtype = Dtype.from_duckdb(relation.types[0])
        if dtype in [types.float32, types.float64]:
            return FloatSqlBackedFeature(sql_query=sql_query,
                                         params=params, secondary_tables=tuple(secondary_tables), join_strategies=(),
                                         primary_table_name=primary_table_name, index_column_name=index_column_name,
                                         name=name, description=description,
                                         technical_description=technical_description,
                                         default_for_missing=default_for_missing, default_for_nan=default_for_nan,
                                         default_for_infinity=default_for_infinity,
                                         default_for_neg_infinity=default_for_neg_infinity)
        else:
            raise ValueError(f'SQL query returned type {dtype} and not float or double')


def categorical_feature_from_query(conn: DuckDBPyConnection,
                                   sql_query: str,
                                   params: Schema, secondary_tables: Sequence[DuckdbTable],

                                   primary_table_name: str = 'primary_table',
                                   index_column_name: str = 'rowid',

                                   name: str | None = None,
                                   description: str = '',
                                   technical_description: str | None = None,

                                   default_for_missing: str = CategoricalFeature.other_category) -> CategoricalSqlBackedFeature:
    """As feature_from_query, but always creates a categorical feature and raises an error if the query returns a different type."""
    with conn.cursor() as cursor:
        relation = _relation_from_query(cursor, sql_query, params, primary_table_name, index_column_name, secondary_tables)

        name = name or relation.columns[0]
        technical_description = technical_description or sql_query

        dtype = Dtype.from_duckdb(relation.types[0])
        if dtype == types.string:
            return CategoricalSqlBackedFeature(sql_query=sql_query,
                                               params=params, secondary_tables=tuple(secondary_tables),
                                               join_strategies=(),
                                               primary_table_name=primary_table_name,
                                               index_column_name=index_column_name,
                                               name=name, description=description,
                                               technical_description=technical_description,
                                               default_for_missing=default_for_missing,
                                               categories=('nonesuch',))
        elif isinstance(dtype, types.EnumDtype):
            return CategoricalSqlBackedFeature(sql_query=sql_query,
                                               params=params, secondary_tables=tuple(secondary_tables),
                                               join_strategies=(),
                                               primary_table_name=primary_table_name,
                                               index_column_name=index_column_name,
                                               name=name, description=description,
                                               technical_description=technical_description,
                                               default_for_missing=default_for_missing,
                                               categories=dtype.values)
        else:
            raise ValueError(f'SQL query returned type {dtype} and not varchar or enum')
