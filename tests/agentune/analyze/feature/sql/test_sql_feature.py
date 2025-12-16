
import attrs
import duckdb
import polars as pl
import pytest
from duckdb import DuckDBPyConnection

from agentune.analyze.feature.base import CategoricalFeature
from agentune.analyze.feature.sql.base import (
    BoolSqlBackedFeature,
    CategoricalSqlBackedFeature,
    FloatSqlBackedFeature,
    IntSqlBackedFeature,
)
from agentune.analyze.feature.validate.base import FeatureValidationError
from agentune.core import types
from agentune.core.database import DuckdbTable
from agentune.core.dataset import Dataset
from agentune.core.schema import Field, Schema


def test_int_feature(conn: DuckDBPyConnection) -> None:
    conn.execute('CREATE TABLE context_table (key int, value int)')
    conn.execute('INSERT INTO context_table VALUES (1, 2), (3, 4)')

    sql_query = '''
                SELECT context_table.value
                FROM main_table
                         LEFT JOIN context_table ON main_table.key = context_table.key
                ORDER BY main_table.rowid
                '''

    context_table = DuckdbTable.from_duckdb('context_table', conn)
    feature = IntSqlBackedFeature(
        sql_query=sql_query,
        primary_table_name='main_table', index_column_name='rowid',
        name='my feature', description='',
        params = Schema((Field('key', types.int32), )),
        secondary_tables=(context_table,),
        join_strategies=(),
        technical_description='',
        default_for_missing=0
    )

    assert feature.compute((1,), conn) == 2
    assert feature.compute((3,), conn) == 4
    assert feature.compute((2,), conn) is None

    batch_input = Dataset(feature.params, pl.DataFrame({'key': [3, 2, 1, 3, 1]}))
    batch_expected_result = pl.Series('my feature', [4, None, 2, 4, 2], dtype=pl.Int32)
    assert feature.compute_batch(batch_input, conn).equals(batch_expected_result, check_names=True, check_dtypes=True)

    for tpe in [types.int16, types.int8, types.uint16, types.uint8]:
        feature = attrs.evolve(feature, sql_query = f'select value::{tpe.duckdb_type} from ({sql_query})')
        assert feature.compute_batch(batch_input, conn).equals(batch_expected_result, check_names=True, check_dtypes=True), \
            'Feature returns different but compatible dtype'

    for tpe in [types.uint32, types.int64, types.float64]:
        feature = attrs.evolve(feature, sql_query = f'select value::{tpe.duckdb_type} from ({sql_query})')
        assert feature.compute_batch(batch_input, conn).equals(batch_expected_result, check_names=True, check_dtypes=True), \
            'Feature returns different dtype but the values can be represented exactly as an int32'

    # Value cannot be represented exactly as an int32; SQL query fails
    feature = attrs.evolve(feature, sql_query = f'select {2**32}::uint32 from main_table')
    with pytest.raises(duckdb.ConversionException):
        feature.compute_batch(batch_input, conn)

    # Value cannot be represented exactly as an int32; SQL query succeeds but polars cast fails
    feature = attrs.evolve(feature, sql_query = f'select {2**32} from main_table')
    with pytest.raises(FeatureValidationError, match='cannot be cast'):
        feature.compute_batch(batch_input, conn)

    # Query returns the wrong number of rows
    feature = attrs.evolve(feature, sql_query = '''
                                                SELECT context_table.value
                                                FROM main_table
                                                         JOIN context_table ON main_table.key = context_table.key
                                                ORDER BY main_table.rowid
                                                ''')
    with pytest.raises(FeatureValidationError, match='wrong number of rows'):
        feature.compute_batch(batch_input, conn)

    # Query returns the wrong number of columns
    feature = attrs.evolve(feature, sql_query='select key as key1, key as key2 from main_table')
    with pytest.raises(FeatureValidationError, match='2 columns instead of one'):
        feature.compute_batch(batch_input, conn)

# Non-int feature tests don't repeat testing functionality that's in the base class e.g. query returning a wrong number
# of rows or columns

def test_float_feature(conn: DuckDBPyConnection) -> None:
    conn.execute('CREATE TABLE context_table (key int, value float)')
    conn.execute('INSERT INTO context_table VALUES (1, 2.5), (3, 4.5)')

    sql_query = '''
                SELECT context_table.value
                FROM main_table
                         LEFT JOIN context_table ON main_table.key = context_table.key
                ORDER BY main_table.rowid
                '''

    context_table = DuckdbTable.from_duckdb('context_table', conn)
    feature = FloatSqlBackedFeature(
        sql_query=sql_query,
        primary_table_name='main_table', index_column_name='rowid',
        name='my feature', description='',
        params = Schema((Field('key', types.int32), )),
        secondary_tables=(context_table,),
        join_strategies=(),
        technical_description='',
        default_for_missing=0.0,
        default_for_nan=0.0,
        default_for_infinity=0.0,
        default_for_neg_infinity=0.0
    )

    assert feature.compute((1,), conn) == 2.5
    assert feature.compute((3,), conn) == 4.5
    assert feature.compute((2,), conn) is None

    batch_input = Dataset(feature.params, pl.DataFrame({'key': [3, 2, 1, 3, 1]}))
    batch_expected_result = pl.Series('my feature', [4.5, None, 2.5, 4.5, 2.5], dtype=pl.Float64)
    assert feature.compute_batch(batch_input, conn).equals(batch_expected_result, check_names=True, check_dtypes=True)

    for tpe in [types.float32]:
        feature = attrs.evolve(feature, sql_query = f'select value::{tpe.duckdb_type} from ({sql_query})')
        assert feature.compute_batch(batch_input, conn).equals(batch_expected_result, check_names=True, check_dtypes=True), \
            'Feature returns different but compatible dtype'

    for tpe in [types.uint32, types.int64, types.uint64]:
        feature = attrs.evolve(feature, sql_query = f'select value::{tpe.duckdb_type} from ({sql_query})')
        # Expect values rounded to ints
        batch_expected_result = pl.Series('my feature', [4.0, None, 2.0, 4.0, 2.0], dtype=pl.Float64)
        assert feature.compute_batch(batch_input, conn).equals(batch_expected_result, check_names=True, check_dtypes=True), \
            'Feature returns different dtype but the values can be represented exactly as a float64'


def test_bool_feature(conn: DuckDBPyConnection) -> None:
    conn.execute('CREATE TABLE context_table (key int, value float)')
    conn.execute('INSERT INTO context_table VALUES (1, 2.5), (3, 4.5)')

    sql_query = '''
                SELECT context_table.value > 3.0
                FROM main_table
                         LEFT JOIN context_table ON main_table.key = context_table.key
                ORDER BY main_table.rowid
                '''

    context_table = DuckdbTable.from_duckdb('context_table', conn)
    feature = BoolSqlBackedFeature(
        sql_query=sql_query,
        primary_table_name='main_table', index_column_name='rowid',
        name='my feature', description='',
        params = Schema((Field('key', types.int32), )),
        secondary_tables=(context_table,),
        join_strategies=(),
        technical_description='',
        default_for_missing=False,
    )

    assert feature.compute((1,), conn) is False
    assert feature.compute((3,), conn) is True
    assert feature.compute((2,), conn) is None

    batch_input = Dataset(feature.params, pl.DataFrame({'key': [3, 2, 1, 3, 1]}))
    batch_expected_result = pl.Series('my feature', [True, None, False, True, False], dtype=pl.Boolean)
    assert feature.compute_batch(batch_input, conn).equals(batch_expected_result, check_names=True, check_dtypes=True)


def test_categorical_feature(conn: DuckDBPyConnection) -> None:
    conn.execute('CREATE TABLE context_table (key int, value int)')
    conn.execute('INSERT INTO context_table VALUES (1, 2), (3, 4)')

    sql_query = f'''
                SELECT context_table.value::string::ENUM('2', '4', '{CategoricalFeature.other_category}') as value
                FROM main_table
                         LEFT JOIN context_table ON main_table.key = context_table.key
                ORDER BY main_table.rowid
                '''

    context_table = DuckdbTable.from_duckdb('context_table', conn)
    feature = CategoricalSqlBackedFeature(
        sql_query=sql_query,
        primary_table_name='main_table', index_column_name='rowid',
        name='my feature', description='',
        params = Schema((Field('key', types.int32), )),
        secondary_tables=(context_table,),
        join_strategies=(),
        technical_description='',
        categories=('2', '4'),
        default_for_missing=CategoricalFeature.other_category
    )

    assert feature.compute((1,), conn) == '2'
    assert feature.compute((3,), conn) == '4'
    assert feature.compute((2,), conn) is None

    batch_input = Dataset(feature.params, pl.DataFrame({'key': [3, 2, 1, 3, 1]}))
    batch_expected_result = pl.Series('my feature', ['4', None, '2', '4', '2'], dtype=pl.Enum(categories=['2', '4', CategoricalFeature.other_category]))
    batch_result = feature.compute_batch(batch_input, conn)
    assert batch_result.equals(batch_expected_result, check_names=True, check_dtypes=True)

    for tpe in [types.string, types.EnumDtype('2', '4', '5')]:
        feature = attrs.evolve(feature, sql_query = f'select value::{tpe.duckdb_type} from ({sql_query})')
        assert feature.compute_batch(batch_input, conn).equals(batch_expected_result, check_names=True, check_dtypes=True), \
            'Feature returns different but compatible dtype'

def test_synthetic_rowid(conn: DuckDBPyConnection) -> None:
    conn.execute('CREATE TABLE main_table (key int)')
    conn.execute('INSERT INTO main_table VALUES (1), (2,)')
    
    assert conn.execute('select rowid from main_table').fetchall() == [(0,), (1,)], 'Native rowid is 0-based'

    sql_query = '''
                SELECT rowid
                FROM main_table
                ORDER BY main_table.rowid
                '''

    main_table = DuckdbTable.from_duckdb('main_table', conn)
    feature = IntSqlBackedFeature(
        sql_query=sql_query,
        primary_table_name='main_table', index_column_name='rowid',
        name='my feature', description='',
        params = Schema((Field('key', types.int32), )),
        secondary_tables=(main_table,),
        join_strategies=(),
        technical_description='',
        default_for_missing=0
    )

    batch_input = Dataset(feature.params, pl.DataFrame({'key': range(10)}))
    batch_expected_result = pl.Series('my feature', range(10), dtype=pl.Int32)
    assert feature.compute_batch(batch_input, conn).equals(batch_expected_result, check_names=True, check_dtypes=True)

