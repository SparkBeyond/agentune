import datetime
import logging
import uuid
from typing import Any

import polars as pl
from duckdb.duckdb import DuckDBPyConnection

from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbManager
from agentune.analyze.core.dataset import Dataset, DatasetSink, DatasetSource, duckdb_to_polars
from agentune.analyze.core.schema import Field, Schema, restore_relation_types

_logger = logging.getLogger(__name__)


def test_types_and_schema(conn: DuckDBPyConnection) -> None:
    """Test that types and values roundtrip between duckdb and polars, and that schema discovery works."""
    all_types = [
        *types._simple_dtypes,
        types.EnumDtype('a', 'b', 'c'),
        types.ListDtype(types.int32),
        types.ListDtype(types.EnumDtype('a', 'b', 'c')),
        types.ArrayDtype(types.int32, 3),
        types.ArrayDtype(types.ArrayDtype(types.EnumDtype('a', 'b'), 5), 3),
        types.StructDtype(('a', types.int32), ('b', types.string)),
        # Enums inside structs make Polars panic
    ]

    for dtype in all_types:
        assert types.Dtype.from_duckdb(dtype.duckdb_type) == dtype
        if dtype is types.json_dtype or dtype is types.uuid_dtype:
            assert types.Dtype.from_polars(dtype.polars_type) == types.string
        else:
            assert types.Dtype.from_polars(dtype.polars_type) == dtype

    dtype_set = set(all_types)
    assert len(dtype_set) == len(all_types), 'Dtypes must be hashable and comparable'

    cols = [f'"col_{dtype.name}" {dtype.duckdb_type}' for dtype in all_types]
    conn.execute(f"create table tab ({', '.join(cols)})")

    relation = conn.table('tab')
    schema = Schema.from_duckdb(relation)
    assert schema.dtypes == all_types

    df = duckdb_to_polars(relation)

    def expected_dtype(dtype: types.Dtype) -> types.Dtype:
        if dtype in (types.json_dtype, types.uuid_dtype):
            return types.string
        else:
            return dtype

    expected_schema = Schema(tuple(Field(col.name, expected_dtype(col.dtype)) for col in schema.cols))

    assert Schema.from_polars(df) == expected_schema

    bad_df = relation.pl()
    assert Schema.from_polars(bad_df) != expected_schema, 'Direct export to polars loses type information'

    relation = conn.from_arrow(df.to_arrow())
    assert Schema.from_duckdb(relation) != expected_schema, 'Duckdb reading dataframe / arrow loses type information'

    fixed_relation = restore_relation_types(relation, expected_schema)
    assert Schema.from_duckdb(fixed_relation) == expected_schema

def test_scalar_types(conn: DuckDBPyConnection) -> None:
    """Test that the declared Python scalar type matches what we get from duckdb and from polars,
    and values roundtrip.
    """
    all_types = [
        *types._simple_dtypes,
        types.EnumDtype('a', 'b', 'c'),
        types.ListDtype(types.int32),
        types.ArrayDtype(types.int32, 3),
        types.StructDtype(('a', types.int32), ('b', types.string)),
    ]

    def value_for_type(dtype: types.Dtype) -> Any:
        match dtype:
            case types.boolean: return True
            case _ if dtype.is_integer(): return 1
            case _ if dtype.is_float(): return 1.0
            case types.string: return 'a'
            case types.json_dtype: return '{"a": 1}'
            case types.uuid_dtype: return uuid.UUID('06100a4d-34ca-430a-bd30-db6371e209e3')
            case types.date_dtype: return datetime.date.today()
            case types.time_dtype: return datetime.time(0)
            case types.timestamp: return datetime.datetime(2000, 1, 2, 3, 4, 5, 1000) # ms precision
            case types.EnumDtype(): return 'a'
            case types.ListDtype(): return [1,2,3]
            case types.ArrayDtype(): return (1,2,3)
            case types.StructDtype(): return {'a': 1, 'b': 'foo'}
            case _: raise ValueError(f'Unsupported type {dtype}')

    cols = [f'"col_{dtype.name}" {dtype.duckdb_type}' for dtype in all_types]
    _logger.info(', '.join(cols))
    conn.execute(f"create table tab ({', '.join(cols)})")
    conn.execute(f'''insert into tab({', '.join([f'"col_{dtype.name}"' for dtype in all_types])}) 
                            values({', '.join(['?' for _dtype in all_types])})''',
                 [value_for_type(dtype) for dtype in all_types])

    duckdb_results = conn.execute('select * from tab').fetchone()
    polars_results = duckdb_to_polars(conn.table('tab')).row(0)
    expected = tuple(value_for_type(dtype) for dtype in all_types)
    for dtype, duckdb_value, polars_value, expected_value in zip(all_types, duckdb_results, polars_results, expected, strict=True):
        assert type(polars_value) is types.python_type_from_polars(dtype), \
            f'Polars value {polars_value} for dtype {dtype} has python type {type(polars_value)}, expected {types.python_type_from_polars(dtype)}'
        assert type(duckdb_value) is types.python_type_from_duckdb(dtype), \
            f'Duckdb value {duckdb_value} for dtype {dtype} has python type {type(duckdb_value)}, expected {types.python_type_from_duckdb(dtype)}'

        # Adjust known expected value differences
        polars_expected_value = expected_value
        if isinstance(dtype, types.ArrayDtype):
            polars_expected_value = list(expected_value)
        elif dtype == types.uuid_dtype:
            polars_expected_value = str(expected_value)

        assert duckdb_value == expected_value, f'Duckdb value {duckdb_value} for {dtype} does not match expected value {expected_value}'
        assert polars_value == polars_expected_value, f'Polars value {polars_value} for {dtype} does not match expected value {polars_expected_value}'


def test_restore_relation_types(ddb_manager: DuckdbManager) -> None:
    enum_type = types.EnumDtype('a', 'b')
    dataset = Dataset.from_polars(
        pl.DataFrame({
            'int': [0, 1, 2, 3, 4, 5],
            'enum': ['a', 'b', 'a', 'b', 'a', 'b'],
        }, schema={
            'int': types.int64.polars_type,
            'enum': enum_type.polars_type
        })
    )

    with ddb_manager.cursor() as conn:
        DatasetSink.into_unqualified_duckdb_table('sink', conn).write(dataset.as_source(), conn)
        reread = DatasetSource.from_table_name('sink', conn).to_dataset(conn)
        assert reread.schema == dataset.schema
        assert reread.data.equals(dataset.data)


