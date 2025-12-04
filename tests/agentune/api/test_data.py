from io import StringIO
from pathlib import Path

import polars as pl
import pytest

from agentune.api.base import RunContext
from agentune.api.data import BoundDatasetSink, BoundDatasetSource
from agentune.core import types
from agentune.core.database import DuckdbName, DuckdbTable
from agentune.core.dataset import (
    Dataset,
    DatasetSource,
    ReadCsvParams,
    ReadNdjsonParams,
    WriteCsvParams,
    WriteParquetParams,
)
from agentune.core.duckdbio import IfTargetExists
from agentune.core.schema import Field, Schema


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return pl.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'value': [10.5, 20.0, 30.5, 40.0, 50.5],
        'category': pl.Series(['A', 'B', 'A', 'C', 'B'], dtype=pl.Enum(['A', 'B', 'C'])),
        'timestamp': pl.Series(['2024-01-01 10:00:00', '2024-01-02 11:00:00', '2024-01-03 12:00:00',
                               '2024-01-04 13:00:00', '2024-01-05 14:00:00']).str.to_datetime(time_unit='ms'),
    })


async def test_wrapped_polars_dataframe(ctx: RunContext, sample_df: pl.DataFrame) -> None:
    source = ctx.data.from_df(sample_df)
    assert isinstance(source, BoundDatasetSource)
    assert source.schema == Schema.from_polars(sample_df)
    assert await source.size() == 5
    assert source.cheap_size == 5

    result_df = (await source.load()).data
    assert result_df.equals(sample_df)


async def test_wrapped_dataset(ctx: RunContext, sample_df: pl.DataFrame) -> None:
    dataset = Dataset.from_polars(sample_df)
    source = ctx.data.from_df(dataset)
    assert source.schema == dataset.schema
    assert await source.size() == 5
    assert (await source.load()).data.equals(sample_df)


async def test_wrapped_dataset_source(ctx: RunContext, sample_df: pl.DataFrame) -> None:
    dataset_source = DatasetSource.from_dataset(Dataset.from_polars(sample_df))
    source = ctx.data.from_df(dataset_source)
    assert source.schema == dataset_source.schema
    assert await source.size() == 5
    assert (await source.load()).data.equals(sample_df)

async def test_sql_with_params(ctx: RunContext, sample_df: pl.DataFrame) -> None:
    await ctx.data.from_df(sample_df).copy_to_table('test_table')
    source = ctx.data.from_sql('SELECT * FROM test_table WHERE id > ?', params=[2])
    expected_df = sample_df.filter(pl.col('id') > 2)
    assert await source.size() == 3
    assert (await source.load()).data.equals(expected_df)

    await ctx.data.from_df(sample_df).copy_to_table('test_table', if_exists=IfTargetExists.APPEND)
    assert await source.size() == 6
    assert (await source.load()).data.equals(expected_df.vstack(expected_df))


async def test_sql_callable(ctx: RunContext, sample_df: pl.DataFrame) -> None:
    await ctx.data.from_df(sample_df).copy_to_table('test_table')
    source = ctx.data.from_sql(lambda _: _.sql('SELECT * FROM test_table WHERE id < 3'))
    expected_df = sample_df.filter(pl.col('id') < 3)
    assert await source.size() == 2
    assert (await source.load()).data.equals(expected_df)


async def test_csv_from_file(ctx: RunContext, tmp_path: Path, sample_df: pl.DataFrame) -> None:
    csv_path = tmp_path / 'test.csv'
    sample_df.write_csv(csv_path)

    source = ctx.data.from_csv(csv_path)
    assert await source.size() == 5
    # CSV loses enum type info - category becomes string
    assert source.schema == Schema.from_polars(sample_df.cast({'category': str}))
    assert (await source.load()).data.equals(sample_df.cast({'category': str}))

    # Use dtype to specify the enum type for the category column
    category_dtype = types.EnumDtype('A', 'B', 'C')
    source = ctx.data.from_csv(csv_path, ReadCsvParams(dtype={'category': str(category_dtype.duckdb_type)}))
    assert source.schema == Schema.from_polars(sample_df)
    assert (await source.load()).data.equals(sample_df)


async def test_csv_from_stringio(ctx: RunContext) -> None:
    csv_content = 'a,b,c\n1,2,3\n4,5,6'
    source = ctx.data.from_csv(StringIO(csv_content))
    assert await source.size() == 2
    assert source.schema == Schema((
        Field('a', types.int64),
        Field('b', types.int64),
        Field('c', types.int64),
    ))

    expected_df = pl.DataFrame({'a': [1, 4], 'b': [2, 5], 'c': [3, 6]})
    result_df = (await source.load()).data
    assert result_df.equals(expected_df)


async def test_csv_with_params(ctx: RunContext, tmp_path: Path) -> None:
    csv_path = tmp_path / 'test.csv'
    csv_path.write_text('a|b|c\n1|2|3\n4|5|6')
    source = ctx.data.from_csv(csv_path, ReadCsvParams(delimiter='|'))
    assert await source.size() == 2
    assert source.schema == Schema((
        Field('a', types.int64),
        Field('b', types.int64),
        Field('c', types.int64),
    ))

    expected_df = pl.DataFrame({'a': [1, 4], 'b': [2, 5], 'c': [3, 6]})
    result_df = (await source.load()).data
    assert result_df.equals(expected_df)


async def test_parquet_from_file(ctx: RunContext, tmp_path: Path, sample_df: pl.DataFrame) -> None:
    parquet_path = tmp_path / 'test.parquet'
    await ctx.data.from_df(sample_df).copy_to(ctx.data.to_parquet(parquet_path))
    source = ctx.data.from_parquet(parquet_path)
    assert await source.size() == sample_df.height

    # Enums written to parquet become strings (aoa#262)
    result_df = (await source.load()).data
    if result_df['category'].dtype == pl.String:
        expected_df = sample_df.with_columns(pl.col('category').cast(pl.String))
        assert result_df.equals(expected_df)
    else:
        assert result_df.equals(sample_df)


async def test_ndjson_from_file(ctx: RunContext, tmp_path: Path) -> None:
    json_path = tmp_path / 'test.json'
    json_path.write_text('{"a":1,"b":2}\n{"a":3,"b":4}')
    source = ctx.data.from_ndjson(json_path)
    assert await source.size() == 2
    assert source.schema == Schema((
        Field('a', types.int64),
        Field('b', types.int64),
    ))

    expected_df = pl.DataFrame({'a': [1, 3], 'b': [2, 4]})
    result_df = (await source.load()).data
    assert result_df.equals(expected_df)


async def test_ndjson_from_stringio(ctx: RunContext) -> None:
    json_content = '{"x":1,"y":2}\n{"x":3,"y":4}'
    source = ctx.data.from_ndjson(StringIO(json_content))
    assert await source.size() == 2
    assert source.schema == Schema((
        Field('x', types.int64),
        Field('y', types.int64),
    ))

    expected_df = pl.DataFrame({'x': [1, 3], 'y': [2, 4]})
    result_df = (await source.load()).data
    assert result_df.equals(expected_df)


async def test_ndjson_with_params(ctx: RunContext, tmp_path: Path) -> None:
    json_path = tmp_path / 'test.json'
    json_path.write_text('[{"a":1},{"a":2}]')
    source = ctx.data.from_ndjson(json_path, ReadNdjsonParams(format='array'))
    assert await source.size() == 2
    assert (await source.load()).data.equals(pl.DataFrame({'a': [1, 2]}))


async def test_to_table(ctx: RunContext, sample_df: pl.DataFrame) -> None:
    sink = ctx.data.to_table('test_table')
    with ctx._ddb_manager.cursor() as conn:
        qualname = DuckdbName.qualify('test_table', conn)
        assert sink == ctx.data.to_table(qualname)
        table = DuckdbTable(qualname, Schema(()), ())
        assert sink == ctx.data.to_table(table)
    assert isinstance(sink, BoundDatasetSink)

    await ctx.data.from_df(sample_df).copy_to(sink)
    assert (await ctx.db.table('test_table').load()).data.equals(sample_df), 'Target table was created'

    await ctx.data.from_df(sample_df).copy_to(ctx.data.to_table('test_table', if_exists=IfTargetExists.APPEND))
    assert (await ctx.db.table('test_table').load()).data.equals(sample_df.vstack(sample_df)), 'Target table was appended'

    await ctx.data.from_df(sample_df).copy_to(sink)
    assert (await ctx.db.table('test_table').load()).data.equals(sample_df), 'Target table was replaced'

    with pytest.raises(ValueError, match='already exists'):
        await ctx.data.from_df(sample_df).copy_to(ctx.data.to_table('test_table', if_exists=IfTargetExists.FAIL))

    await ctx.db.execute('drop table test_table')
    with pytest.raises(ValueError, match='does not exist'):
        await ctx.data.from_df(sample_df).copy_to(ctx.data.to_table('test_table', create_if_not_exists=False))


async def test_to_csv(ctx: RunContext, tmp_path: Path, sample_df: pl.DataFrame) -> None:
    csv_path = tmp_path / 'output.csv'
    await ctx.data.from_df(sample_df).copy_to(ctx.data.to_csv(csv_path))
    assert csv_path.exists()

    result_source = ctx.data.from_csv(csv_path)
    assert await result_source.size() == 5
    # CSV loses enum type, compare with category cast to string
    expected_df = sample_df.with_columns(pl.col('category').cast(pl.String))
    assert (await result_source.load()).data.equals(expected_df)


async def test_to_csv_with_params(ctx: RunContext, tmp_path: Path, sample_df: pl.DataFrame) -> None:
    csv_path = tmp_path / 'output.csv'
    await ctx.data.from_df(sample_df).copy_to(ctx.data.to_csv(csv_path, WriteCsvParams(sep='|')))
    expected_df = sample_df.with_columns(pl.col('category').cast(pl.String))
    assert not (await ctx.data.from_csv(csv_path, ReadCsvParams(sep=',')).load()).data.equals(expected_df)
    assert (await ctx.data.from_csv(csv_path, ReadCsvParams(sep='|')).load()).data.equals(expected_df)


async def test_to_parquet(ctx: RunContext, tmp_path: Path, sample_df: pl.DataFrame) -> None:
    parquet_path = tmp_path / 'output.parquet'
    await ctx.data.from_df(sample_df).copy_to(ctx.data.to_parquet(parquet_path))
    assert parquet_path.exists()

    result_source = ctx.data.from_parquet(parquet_path)
    assert await result_source.size() == 5
    result_df = (await result_source.load()).data
    # Parquet may lose enum type (aoa#262)
    expected_df = sample_df.with_columns(pl.col('category').cast(pl.String)) if result_df['category'].dtype == pl.String else sample_df
    assert result_df.equals(expected_df)

    file_size = parquet_path.stat().st_size
    await ctx.data.from_df(sample_df).copy_to(ctx.data.to_parquet(parquet_path, WriteParquetParams(compression='gzip')))
    new_file_size = parquet_path.stat().st_size
    assert new_file_size > file_size, 'gzip bigger than snappy (parameter took effect)'


async def test_bound_dataset_source_sample_as_string(ctx: RunContext, sample_df: pl.DataFrame) -> None:
    source = ctx.data.from_df(sample_df)
    sample_str = source.sample_as_string(3)
    assert 'Alice' in sample_str
    assert 'Bob' in sample_str
    assert 'Charlie' in sample_str


async def test_bound_dataset_source_copy_to_different_context_raises(ctx: RunContext, sample_df: pl.DataFrame) -> None:
    source = ctx.data.from_df(sample_df)
    async with await RunContext.create() as ctx2:
        sink = ctx2.data.to_table('other_table')
        with pytest.raises(ValueError, match='different RunContext'):
            await source.copy_to(sink)


async def test_bound_table_split(ctx: RunContext) -> None:
    df = pl.DataFrame({'id': range(10000), 'val': range(10000)})
    table = await ctx.data.from_df(df).copy_to_table('test_table')
    split = await table.split(feature_search_size=100, feature_eval_size=200)

    assert split.table.name.name == 'test_table'
    assert split.name.name == 'test_table'
    assert '_is_train' in split.splits.table.schema.names
    assert '_is_feature_search' in split.splits.table.schema.names
    assert '_is_feature_eval' in split.splits.table.schema.names

    train_size = await split.train.size()
    test_size = await split.test.size()
    assert train_size + test_size == 10000
    assert 7990 <= train_size <= 8010  # ~80% train, only accurate to 1% precision and sometimes much worse than that, aoa#263
    assert await split.feature_search.size() == 100
    assert await split.feature_eval.size() == 200


async def test_bound_table_split_if_not_exists(ctx: RunContext) -> None:
    df = pl.DataFrame({'id': range(100), 'val': range(100)})
    table = await ctx.data.from_df(df).copy_to_table('test_table')
    split_table = await table.split(if_not_exists=True)

    with pytest.raises(ValueError, match='already exist'):
        # NOTE we can't call table.split again because `table` remembers the previous non-split schema.
        # This is a general problem with the API design
        await ctx.db.table('test_table').split(if_not_exists=False)

    table_contents = (await split_table.load()).data

    await ctx.db.execute('update test_table set _is_train = false')
    table_contents2 = (await split_table.load()).data
    assert not table_contents2.equals(table_contents), 'Data updated'
    await ctx.db.table(split_table.name).split(if_not_exists=True)
    assert (await split_table.load()).data.equals(table_contents2), 'Did not re-split the table'

