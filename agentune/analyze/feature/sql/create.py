import datetime
import enum
from collections.abc import Sequence

import polars as pl
from duckdb import BinderException, DuckDBPyConnection, DuckDBPyRelation

from agentune.analyze.feature.base import (
    CategoricalFeature,
)
from agentune.analyze.feature.validate.base import FeatureValidationCode, FeatureValidationError
from agentune.core import types
from agentune.core.database import DuckdbTable
from agentune.core.dataset import Dataset
from agentune.core.schema import Schema
from agentune.core.types import Dtype

from .base import (
    BoolSqlBackedFeature,
    CategoricalSqlBackedFeature,
    FloatSqlBackedFeature,
    IntSqlBackedFeature,
    SqlBackedFeature,
    SqlFeatureSpec,
    _register_input_table_with_index,
)


class QueryValidationCode(FeatureValidationCode):
    shadows_index_column_name = enum.auto()
    shadows_main_table_name = enum.auto()
    illegal = enum.auto()
    width = enum.auto()
    dtype = enum.auto()


def _relation_from_query(conn: DuckDBPyConnection,
                         sql_query: str,
                         params: Schema,
                         primary_table_name: str,
                         index_column_name: str,
                         secondary_tables: Sequence[DuckdbTable]) -> DuckDBPyRelation:
    if index_column_name in params.names:
        raise FeatureValidationError(QueryValidationCode.shadows_index_column_name,
                                     f'Input data already has a column named {index_column_name}')
    if primary_table_name in [table.name.name for table in secondary_tables]:
        raise FeatureValidationError(QueryValidationCode.shadows_main_table_name,
                                     f"Primary table name {primary_table_name} shadows secondary table's local name")

    dataset = Dataset(params,
                      pl.DataFrame({
                          field.name: pl.Series(field.name, [], field.dtype.polars_type) for field in params.cols
                      }))
    _register_input_table_with_index(conn, dataset, primary_table_name, index_column_name)
    try:
        relation = conn.sql(sql_query)
    except BinderException as e:
        raise FeatureValidationError(QueryValidationCode.illegal,
                                     f'Illegal query: {e}') from e

    if len(relation.types) != 1:
        raise FeatureValidationError(QueryValidationCode.width,
            f'SQL query must return exactly one column but returned {len(relation.types)}: {relation.columns}')

    return relation


def feature_from_query(conn: DuckDBPyConnection,
                       sql_feature_spec: SqlFeatureSpec,
                       params: Schema,
                       secondary_tables: Sequence[DuckdbTable],
                       primary_table_name: str = 'primary_table',
                       index_column_name: str = 'rowid',
                       timeout: datetime.timedelta | None = None) -> SqlBackedFeature:
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
         sql_feature_spec: specification of the SQL feature to create:
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
            name: name of the created feature. If None, uses the name of the query's output column.
            description: populates Feature.description.
            technical_description: populates Feature.technical_description. If None, set to the query string.
        params: expected schema of the primary input table, which will be available to the query under the name primary_table_name.
        secondary_tables: names and schemas of the secondary input tales.
        primary_table_name: the name used by the query to refer to the primary input table. This table is not expected
                            to exist in `conn` and it will not be used if it does exist.
        index_column_name: name of a synthetic column with row indexes which will be added to the primary table.
                           The query needs to order the results by this column. May not shadow the name of a preexisting
                           column.
        
    """
    with conn.cursor() as cursor:
        relation = _relation_from_query(cursor, sql_feature_spec.sql_query, params, primary_table_name, index_column_name, secondary_tables)

        name = sql_feature_spec.name or relation.columns[0]
        technical_description = sql_feature_spec.technical_description or sql_feature_spec.sql_query

        dtype = Dtype.from_duckdb(relation.types[0])
        if dtype == types.boolean:
            return BoolSqlBackedFeature(sql_query=sql_feature_spec.sql_query,
                                        params=params, secondary_tables=tuple(secondary_tables), join_strategies=(),
                                        primary_table_name=primary_table_name, index_column_name=index_column_name,
                                        name=name, description=sql_feature_spec.description, technical_description=technical_description,
                                        default_for_missing=False, timeout=timeout)
        elif dtype in [types.float32, types.float64]:
            return FloatSqlBackedFeature(sql_query=sql_feature_spec.sql_query,
                                         params=params, secondary_tables=tuple(secondary_tables), join_strategies=(),
                                         primary_table_name=primary_table_name, index_column_name=index_column_name,
                                         name=name, description=sql_feature_spec.description,
                                         technical_description=technical_description,
                                         default_for_missing=0.0, default_for_nan=0.0, default_for_infinity=0.0,
                                         default_for_neg_infinity=0.0, timeout=timeout)
        elif dtype in [types.int8, types.uint8, types.int16, types.uint16, types.int32, types.uint32, types.int64]:
            return IntSqlBackedFeature(sql_query=sql_feature_spec.sql_query,
                                       params=params, secondary_tables=tuple(secondary_tables), join_strategies=(),
                                       primary_table_name=primary_table_name, index_column_name=index_column_name,
                                       name=name, description=sql_feature_spec.description,
                                       technical_description=technical_description,
                                       default_for_missing=0, timeout=timeout)
        elif dtype == types.string:
            return CategoricalSqlBackedFeature(sql_query=sql_feature_spec.sql_query,
                                               params=params, secondary_tables=tuple(secondary_tables),
                                               join_strategies=(),
                                               primary_table_name=primary_table_name,
                                               index_column_name=index_column_name,
                                               name=name, description=sql_feature_spec.description,
                                               technical_description=technical_description,
                                               default_for_missing=CategoricalFeature.other_category,
                                               categories=('nonesuch',), timeout=timeout)
        elif isinstance(dtype, types.EnumDtype):
            return CategoricalSqlBackedFeature(sql_query=sql_feature_spec.sql_query,
                                               params=params, secondary_tables=tuple(secondary_tables),
                                               join_strategies=(),
                                               primary_table_name=primary_table_name,
                                               index_column_name=index_column_name,
                                               name=name, description=sql_feature_spec.description,
                                               technical_description=technical_description,
                                               default_for_missing=CategoricalFeature.other_category,
                                               categories=dtype.values, timeout=timeout)
        else:
            raise FeatureValidationError(QueryValidationCode.dtype,
                                         f'SQL query returned unsupported type {dtype.duckdb_type}')


def int_feature_from_query(conn: DuckDBPyConnection,
                           sql_feature_spec: SqlFeatureSpec,
                           params: Schema,
                           secondary_tables: Sequence[DuckdbTable],
                           primary_table_name: str = 'primary_table',
                           index_column_name: str = 'rowid',
                           timeout: datetime.timedelta | None = None,

                           default_for_missing: int = 0) -> IntSqlBackedFeature:
    """As feature_from_query, but always creates an Int feature and raises an error if the query returns a different type."""
    with conn.cursor() as cursor:
        relation = _relation_from_query(cursor, sql_feature_spec.sql_query, params, primary_table_name, index_column_name, secondary_tables)

        name = sql_feature_spec.name or relation.columns[0]
        technical_description = sql_feature_spec.technical_description or sql_feature_spec.sql_query

        dtype = Dtype.from_duckdb(relation.types[0])
        if dtype in [types.int8, types.uint8, types.int16, types.uint16, types.int32, types.uint32, types.int64]:
            return IntSqlBackedFeature(sql_query=sql_feature_spec.sql_query,
                                       params=params, secondary_tables=tuple(secondary_tables), join_strategies=(),
                                       primary_table_name=primary_table_name, index_column_name=index_column_name,
                                       name=name, description=sql_feature_spec.description,
                                       technical_description=technical_description,
                                       default_for_missing=default_for_missing,
                                       timeout=timeout)
        else:
            raise FeatureValidationError(QueryValidationCode.dtype,
                                         f'SQL query returned type {dtype.duckdb_type} and not an integer type')


def bool_feature_from_query(conn: DuckDBPyConnection,
                            sql_feature_spec: SqlFeatureSpec,
                            params: Schema,
                            secondary_tables: Sequence[DuckdbTable],
                            primary_table_name: str = 'primary_table',
                            index_column_name: str = 'rowid',
                            timeout: datetime.timedelta | None = None,

                            default_for_missing: bool = False) -> BoolSqlBackedFeature:
    """As feature_from_query, but always creates a boolean feature and raises an error if the query returns a different type."""
    with conn.cursor() as cursor:
        relation = _relation_from_query(cursor, sql_feature_spec.sql_query, params, primary_table_name, index_column_name, secondary_tables)
        name = sql_feature_spec.name or relation.columns[0]
        technical_description = sql_feature_spec.technical_description or sql_feature_spec.sql_query

        dtype = Dtype.from_duckdb(relation.types[0])
        if dtype == types.boolean:
            return BoolSqlBackedFeature(sql_query=sql_feature_spec.sql_query,
                                        params=params, secondary_tables=tuple(secondary_tables), join_strategies=(),
                                        primary_table_name=primary_table_name, index_column_name=index_column_name,
                                        name=name, description=sql_feature_spec.description, technical_description=technical_description,
                                        timeout=timeout,
                                        default_for_missing=default_for_missing)
        else:
            raise FeatureValidationError(QueryValidationCode.dtype,
                                         f'SQL query returned type {dtype.duckdb_type} and not boolean')


def float_feature_from_query(conn: DuckDBPyConnection,
                             sql_feature_spec: SqlFeatureSpec,
                             params: Schema,
                             secondary_tables: Sequence[DuckdbTable],
                             primary_table_name: str = 'primary_table',
                             index_column_name: str = 'rowid',
                             timeout: datetime.timedelta | None = None,

                             default_for_missing: float = 0.0,
                             default_for_nan: float = 0.0,
                             default_for_infinity: float = 0.0,
                             default_for_neg_infinity: float = 0.0) -> FloatSqlBackedFeature:
    """As feature_from_query, but always creates a float feature and raises an error if the query returns a different type."""
    with conn.cursor() as cursor:
        relation = _relation_from_query(cursor, sql_feature_spec.sql_query, params, primary_table_name, index_column_name, secondary_tables)

        name = sql_feature_spec.name or relation.columns[0]
        technical_description = sql_feature_spec.technical_description or sql_feature_spec.sql_query

        dtype = Dtype.from_duckdb(relation.types[0])
        if dtype in [types.float32, types.float64]:
            return FloatSqlBackedFeature(sql_query=sql_feature_spec.sql_query,
                                         params=params, secondary_tables=tuple(secondary_tables), join_strategies=(),
                                         primary_table_name=primary_table_name, index_column_name=index_column_name,
                                         name=name, description=sql_feature_spec.description,
                                         technical_description=technical_description, timeout=timeout,
                                         default_for_missing=default_for_missing, default_for_nan=default_for_nan,
                                         default_for_infinity=default_for_infinity,
                                         default_for_neg_infinity=default_for_neg_infinity)
        else:
            raise FeatureValidationError(QueryValidationCode.dtype,
                                         f'SQL query returned type {dtype.duckdb_type} and not float or double')


def categorical_feature_from_query(conn: DuckDBPyConnection,
                                   sql_feature_spec: SqlFeatureSpec,
                                   params: Schema,
                                   secondary_tables: Sequence[DuckdbTable],
                                   primary_table_name: str = 'primary_table',
                                   index_column_name: str = 'rowid',
                                   timeout: datetime.timedelta | None = None,

                                   default_for_missing: str = CategoricalFeature.other_category) -> CategoricalSqlBackedFeature:
    """As feature_from_query, but always creates a categorical feature and raises an error if the query returns a different type."""
    with conn.cursor() as cursor:
        relation = _relation_from_query(cursor, sql_feature_spec.sql_query, params, primary_table_name, index_column_name, secondary_tables)
        name = sql_feature_spec.name or relation.columns[0]
        technical_description = sql_feature_spec.technical_description or sql_feature_spec.sql_query

        dtype = Dtype.from_duckdb(relation.types[0])
        if dtype == types.string:
            return CategoricalSqlBackedFeature(sql_query=sql_feature_spec.sql_query,
                                               params=params, secondary_tables=tuple(secondary_tables),
                                               join_strategies=(),
                                               primary_table_name=primary_table_name,
                                               index_column_name=index_column_name,
                                               name=name, description=sql_feature_spec.description,
                                               technical_description=technical_description, timeout=timeout,
                                               default_for_missing=default_for_missing,
                                               categories=('nonesuch',))
        elif isinstance(dtype, types.EnumDtype):
            return CategoricalSqlBackedFeature(sql_query=sql_feature_spec.sql_query,
                                               params=params, secondary_tables=tuple(secondary_tables),
                                               join_strategies=(),
                                               primary_table_name=primary_table_name,
                                               index_column_name=index_column_name,
                                               name=name, description=sql_feature_spec.description,
                                               technical_description=technical_description, timeout=timeout,
                                               default_for_missing=default_for_missing,
                                               categories=dtype.values)
        else:
            raise FeatureValidationError(QueryValidationCode.dtype,
                                         f'SQL query returned type {dtype.duckdb_type} and not varchar or enum')
