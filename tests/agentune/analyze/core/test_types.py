
import logging

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
        types.ArrayDtype(types.int32, 3),
        types.StructDtype(('a', types.int32), ('b', types.string))
    ]

    for dtype in all_types:
        assert types.Dtype.from_duckdb(dtype.duckdb_type) == dtype
        if dtype is types.json or dtype is types.uuid:
            assert types.Dtype.from_polars(dtype.polars_type) == types.string
        else:
            assert types.Dtype.from_polars(dtype.polars_type) == dtype

    cols = [f'"col_{dtype.name}" {dtype.duckdb_type}' for dtype in all_types]
    conn.execute(f"create table tab ({', '.join(cols)})")

    relation = conn.table('tab')
    schema = Schema.from_duckdb(relation)
    assert schema.dtypes == all_types

    df = duckdb_to_polars(relation)

    def expected_dtype(dtype: types.Dtype) -> types.Dtype:
        if dtype in (types.json, types.uuid):
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
        DatasetSink.into_duckdb_table('sink').write(dataset.as_source(), conn)
        reread = DatasetSource.from_table_name('sink', conn).to_dataset(conn)
        assert reread.schema == dataset.schema
        assert reread.data.equals(dataset.data)


