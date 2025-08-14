import tempfile
from collections.abc import Callable, Iterator
from io import StringIO
from pathlib import Path

import polars as pl
import pytest
from duckdb import DuckDBPyConnection, DuckDBPyRelation

from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbName
from agentune.analyze.core.dataset import Dataset, DatasetSink, DatasetSource
from agentune.analyze.core.schema import Field, Schema


@pytest.fixture
def sample_dataset() -> Dataset:
    """Create a small sample dataset for testing."""
    data = pl.DataFrame(
        {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000.0, 60000.0, 75000.0, 55000.0, 62000.0],
        }
    )
    return Dataset.from_polars(data)


@pytest.fixture
def large_dataset() -> Dataset:
    """Create a larger dataset for testing chunked reading."""
    size = 50000
    data = pl.DataFrame(
        {
            'id': range(size),
            'value': [i * 2.5 for i in range(size)],
            'category': [f'cat_{i % 10}' for i in range(size)],
        }
    )
    return Dataset.from_polars(data)


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Provide a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_dataset_from_csv_basic(conn: DuckDBPyConnection, temp_dir: Path) -> None:
    """Test basic CSV reading functionality and kwargs passthrough."""
    csv_path = temp_dir / 'test.csv'
    csv_content = 'id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,300'
    csv_path.write_text(csv_content)

    source = DatasetSource.from_csv(csv_path, conn)
    dataset = source.to_dataset(conn)

    assert len(dataset) == 3
    assert dataset.data['id'].to_list() == [1, 2, 3]
    assert dataset.data['name'].to_list() == ['Alice', 'Bob', 'Charlie']
    assert dataset.data['value'].to_list() == [100, 200, 300]

    csv_path_pipe = temp_dir / 'test_pipe.csv'
    csv_content_pipe = 'id|name|value\n1|Alice|100\n2|Bob|200'
    csv_path_pipe.write_text(csv_content_pipe)

    source_pipe = DatasetSource.from_csv(csv_path_pipe, conn, delimiter='|')
    dataset_pipe = source_pipe.to_dataset(conn)

    assert len(dataset_pipe) == 2
    assert dataset_pipe.data['name'].to_list() == ['Alice', 'Bob']


def test_dataset_from_csv_string(conn: DuckDBPyConnection) -> None:
    """Test CSV reading from string data instead of file."""
    csv_content = 'product,price,category\nLaptop,999.99,Electronics\nChair,149.50,Furniture\nBook,12.99,Education'
    
    def make_csv_opener() -> Callable[[DuckDBPyConnection], DuckDBPyRelation]:
        return lambda conn: conn.read_csv(StringIO(csv_content))
    
    source = DatasetSource.from_duckdb_parser(make_csv_opener(), conn)
    dataset = source.to_dataset(conn)
    
    assert len(dataset) == 3
    assert dataset.data['product'].to_list() == ['Laptop', 'Chair', 'Book']
    assert dataset.data['price'].to_list() == [999.99, 149.50, 12.99]
    assert dataset.data['category'].to_list() == ['Electronics', 'Furniture', 'Education']


def test_dataset_from_parquet_basic(conn: DuckDBPyConnection, temp_dir: Path, sample_dataset: Dataset) -> None:
    """Test basic Parquet reading functionality and kwargs passthrough."""
    parquet_path = temp_dir / 'test.parquet'
    sample_dataset.data.write_parquet(parquet_path)

    source = DatasetSource.from_parquet(parquet_path, conn)
    dataset = source.to_dataset(conn)

    assert len(dataset) == 5
    assert dataset.data['id'].to_list() == [1, 2, 3, 4, 5]
    assert dataset.data['name'].to_list() == ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']
    assert dataset.data['age'].to_list() == [25, 30, 35, 28, 32]
    assert dataset.data['salary'].to_list() == [50000.0, 60000.0, 75000.0, 55000.0, 62000.0]

    source_with_kwargs = DatasetSource.from_parquet(parquet_path, conn, filename=True)
    dataset_with_kwargs = source_with_kwargs.to_dataset(conn)

    assert len(dataset_with_kwargs) == 5


def test_dataset_from_json_basic(conn: DuckDBPyConnection, temp_dir: Path) -> None:
    """Test basic JSON reading functionality and kwargs passthrough."""
    json_path = temp_dir / 'test.json'
    json_content = '''[
        {"id": 1, "name": "Alice", "active": true},
        {"id": 2, "name": "Bob", "active": false},
        {"id": 3, "name": "Charlie", "active": true}
    ]'''
    json_path.write_text(json_content)

    source = DatasetSource.from_json(json_path, conn)
    dataset = source.to_dataset(conn)

    assert len(dataset) == 3
    assert dataset.data['id'].to_list() == [1, 2, 3]
    assert dataset.data['name'].to_list() == ['Alice', 'Bob', 'Charlie']
    assert dataset.data['active'].to_list() == [True, False, True]

    source_with_format = DatasetSource.from_json(json_path, conn, format='array')
    dataset_with_format = source_with_format.to_dataset(conn)

    assert len(dataset_with_format) == 3

def test_dataset_into_csv(conn: DuckDBPyConnection, temp_dir: Path, sample_dataset: Dataset) -> None:
    """Test CSV writing functionality and kwargs passthrough."""
    csv_path = temp_dir / 'output.csv'

    sink = DatasetSink.into_csv(csv_path)
    sink.write(sample_dataset.as_source(), conn)
    assert csv_path.exists()
    
    source = DatasetSource.from_csv(csv_path, conn)
    roundtrip_dataset = source.to_dataset(conn)
    
    assert len(roundtrip_dataset) == len(sample_dataset), 'CSV roundtrip length mismatch'
    assert roundtrip_dataset.data.equals(sample_dataset.data), 'CSV roundtrip data mismatch'

    csv_path_pipe = temp_dir / 'output_pipe.csv'
    sink_pipe = DatasetSink.into_csv(csv_path_pipe, sep='|')
    sink_pipe.write(sample_dataset.as_source(), conn)
    assert csv_path_pipe.exists()
    
    source_pipe = DatasetSource.from_csv(csv_path_pipe, conn, delimiter='|')
    roundtrip_pipe_dataset = source_pipe.to_dataset(conn)
    
    assert len(roundtrip_pipe_dataset) == len(sample_dataset), 'Pipe-separated CSV roundtrip length mismatch'
    assert roundtrip_pipe_dataset.data.equals(sample_dataset.data), 'Pipe-separated CSV roundtrip data mismatch'


def test_dataset_into_parquet(conn: DuckDBPyConnection, temp_dir: Path, sample_dataset: Dataset) -> None:
    """Test Parquet writing functionality and kwargs passthrough."""
    parquet_path = temp_dir / 'output.parquet'

    sink = DatasetSink.into_parquet(parquet_path)
    sink.write(sample_dataset.as_source(), conn)
    assert parquet_path.exists()

    source = DatasetSource.from_parquet(parquet_path, conn)
    roundtrip_dataset = source.to_dataset(conn)
    
    assert len(roundtrip_dataset) == len(sample_dataset), 'Parquet roundtrip length mismatch'
    assert roundtrip_dataset.data.equals(sample_dataset.data), 'Parquet roundtrip data mismatch'

    parquet_path_compressed = temp_dir / 'output_compressed.parquet'
    sink_compressed = DatasetSink.into_parquet(parquet_path_compressed, compression='gzip')
    sink_compressed.write(sample_dataset.as_source(), conn)
    assert parquet_path_compressed.exists()
    
    source_compressed = DatasetSource.from_parquet(parquet_path_compressed, conn)
    roundtrip_compressed_dataset = source_compressed.to_dataset(conn)
    
    assert len(roundtrip_compressed_dataset) == len(sample_dataset), 'Compressed Parquet roundtrip length mismatch'
    assert roundtrip_compressed_dataset.data.equals(sample_dataset.data), 'Compressed Parquet roundtrip data mismatch'


def test_dataset_into_duckdb_table(conn: DuckDBPyConnection, sample_dataset: Dataset) -> None:
    """Test writing to DuckDB table with different modes."""
    table_name = DuckdbName.qualify('output_table', conn)

    sink_create = DatasetSink.into_duckdb_table(table_name)
    sink_create.write(sample_dataset.as_source(), conn)

    result = conn.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()
    assert result is not None
    assert result[0] == 5, 'Table should contain 5 rows after creation'

    source = DatasetSource.from_table_name(table_name, conn)
    roundtrip_dataset = source.to_dataset(conn)
    
    assert len(roundtrip_dataset) == len(sample_dataset), 'Table roundtrip length mismatch'
    assert roundtrip_dataset.data.equals(sample_dataset.data), 'Table roundtrip data mismatch'

    sink_append = DatasetSink.into_duckdb_table(
        table_name, create_table=False, delete_contents=False
    )
    sink_append.write(sample_dataset.as_source(), conn)

    result_after_append = conn.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()
    assert result_after_append is not None
    assert result_after_append[0] == 10, 'Table should contain 10 rows after append'
    
    source_after_append = DatasetSource.from_table_name(table_name, conn)
    append_dataset = source_after_append.to_dataset(conn)
    assert len(append_dataset) == 10, 'Appended dataset should have 10 rows'
    
    name_counts = {name: append_dataset.data['name'].to_list().count(name) for name in ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']}
    assert all(count == 2 for count in name_counts.values()), 'Each name should appear twice after append'

    sink_replace = DatasetSink.into_duckdb_table(
        table_name, create_table=False, delete_contents=True
    )
    sink_replace.write(sample_dataset.as_source(), conn)

    result_after_replace = conn.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()
    assert result_after_replace is not None
    assert result_after_replace[0] == 5, 'Table should contain 5 rows after replace'
    
    source_after_replace = DatasetSource.from_table_name(table_name, conn)
    replace_dataset = source_after_replace.to_dataset(conn)
    assert len(replace_dataset) == len(sample_dataset), 'Replace dataset length mismatch'
    assert replace_dataset.data.equals(sample_dataset.data), 'Replace dataset data mismatch'


def test_dataset_from_table_with_batch_size(conn: DuckDBPyConnection) -> None:
    """Test reading from DuckDB table with custom batch_size."""
    table_name = 'test_large_table'
    size = 25000
    conn.execute(f'''
        CREATE TABLE {table_name} AS 
        SELECT 
            row_number() OVER () as id,
            'item_' || row_number() OVER () as name,
            random() * 1000 as value
        FROM generate_series(1, {size})
    ''')

    source_default = DatasetSource.from_table_name(table_name, conn)
    dataset_default = source_default.to_dataset(conn)
    assert len(dataset_default) == size

    batch_size = 5000
    source_custom = DatasetSource.from_table_name(table_name, conn, batch_size=batch_size)
    dataset_custom = source_custom.to_dataset(conn)
    assert len(dataset_custom) == size

    chunks = list(source_custom.open(conn))
    assert len(chunks) > 1, 'Should create multiple chunks'
    assert len(chunks) == (size + batch_size - 1) // batch_size, 'Expected chunk count'

    for i, chunk in enumerate(chunks[:-1]):
        assert len(chunk) == batch_size, f'Chunk {i} should be full size'

    assert len(chunks[-1]) <= batch_size, 'Last chunk should have remaining rows'

    total_rows = sum(len(chunk) for chunk in chunks)
    assert total_rows == size, 'Total rows should match original dataset'

def test_large_data_chunked_reading(conn: DuckDBPyConnection, large_dataset: Dataset) -> None:
    """Test chunked reading with large dataset to verify batch_size works correctly."""
    table_name = 'large_table'
    conn.register('large_temp_data', large_dataset.data.to_arrow())
    conn.execute(f'CREATE TABLE {table_name} AS SELECT * FROM large_temp_data')

    batch_size = 1000
    source = DatasetSource.from_table_name(table_name, conn, batch_size=batch_size)

    chunks = list(source.open(conn))
    assert len(chunks) > 1, 'Should create multiple chunks'

    total_rows = sum(len(chunk) for chunk in chunks)
    assert total_rows == 50000, 'Total rows should match original dataset'

    for chunk in chunks[:-1]:
        assert len(chunk) == batch_size, 'All but last chunk should be full size'

    assert len(chunks[-1]) <= batch_size, 'Last chunk should have remaining rows'

    full_dataset = source.to_dataset(conn)
    assert len(full_dataset) == 50000, 'Full dataset should have correct size'

    first_chunk_ids = chunks[0].data['id'].to_list()
    full_dataset_ids = full_dataset.data['id'].head(len(first_chunk_ids)).to_list()
    assert first_chunk_ids == full_dataset_ids, 'Chunk and full dataset should have consistent data'

def test_cast() -> None:
    data = pl.DataFrame(
        {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'salary': [1.5, 7.8, 9.10, 1000.0, None],
        }
    )
    dataset = Dataset.from_polars(data)

    dataset2 = dataset.cast({'id': types.float32, 'name': types.EnumDtype('Alice', 'Bob', 'Charlie', 'Diana', 'Eve')})
    assert dataset2.schema == Schema(
        (
            Field('id', types.float32),
            Field('name', types.EnumDtype('Alice', 'Bob', 'Charlie', 'Diana', 'Eve')),
            Field('salary', types.float64),
        )
    )
    assert dataset2.data.equals(pl.DataFrame(
        {
            'id': [1.0, 2.0, 3.0, 4.0, 5.0],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'salary': [1.5, 7.8, 9.10, 1000.0, None],
        },
        schema = {
            'id': types.float32.polars_type,
            'name': types.EnumDtype('Alice', 'Bob', 'Charlie', 'Diana', 'Eve').polars_type,
            'salary': types.float64.polars_type,
        }
    ))

def test_cast_no_such_column(sample_dataset: Dataset) -> None:
    with pytest.raises(ValueError, match='not in schema'):
        sample_dataset.cast({'id2': types.float32})

def test_cast_invalid_values(sample_dataset: Dataset) -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError, match=r'conversion from \`str\` to \`enum\` failed'):
        sample_dataset.cast({'name': types.EnumDtype('Alice', 'Bob', 'Charlie')})

    with pytest.raises(pl.exceptions.InvalidOperationError, match=r'conversion from \`str\` to \`i32\` failed'):
        sample_dataset.cast({'name': types.int32})

def test_cast_nonstrict(sample_dataset: Dataset) -> None:
    dataset = sample_dataset.cast({'name': types.float32}, strict=False)
    assert dataset.schema['name'] == Field('name', types.float32)
    assert dataset.data['name'].to_list() == [None, None, None, None, None]

    dataset2 = sample_dataset.cast({'name': types.EnumDtype('Alice', 'Bob', 'Charlie')}, strict=False)
    assert dataset2.data['name'].to_list() == [ 'Alice', 'Bob', 'Charlie', None, None ]

def test_from_csv_basic(conn: DuckDBPyConnection) -> None:
    """Test basic CSV reading from string data."""
    csv_content = '''id,name,value
1,Alice,100.5
2,Bob,200.0
3,Charlie,300.75'''
    csv_io = StringIO(csv_content)
    
    source = DatasetSource.from_csv(csv_io, conn)
    dataset = source.to_dataset(conn)
    
    assert len(dataset) == 3
    assert dataset.data['id'].to_list() == [1, 2, 3]
    assert dataset.data['name'].to_list() == ['Alice', 'Bob', 'Charlie']
    assert dataset.data['value'].to_list() == [100.5, 200.0, 300.75]


def test_from_csv_separator_override(conn: DuckDBPyConnection) -> None:
    """Test CSV reading with custom separator."""
    csv_content = '''id|name|value
1|Alice|100.5
2|Bob|200.0
3|Charlie|300.75'''
    csv_io = StringIO(csv_content)
    
    source = DatasetSource.from_csv(csv_io, conn, delimiter='|')
    dataset = source.to_dataset(conn)
    
    assert len(dataset) == 3
    assert dataset.data['id'].to_list() == [1, 2, 3]
    assert dataset.data['name'].to_list() == ['Alice', 'Bob', 'Charlie']
    assert dataset.data['value'].to_list() == [100.5, 200.0, 300.75]


def test_from_csv_separator_mismatch(conn: DuckDBPyConnection) -> None:
    """Test CSV reading failure when separator doesn't match the data."""
    csv_content = '''id,name,value
1,Alice,100.5
2,Bob,200.0
3,Charlie,300.75'''
    csv_io = StringIO(csv_content)
    
    source = DatasetSource.from_csv(csv_io, conn, delimiter='|')
    dataset = source.to_dataset(conn)
    
    # With wrong delimiter, should parse as single column
    assert len(dataset) == 3
    assert len(dataset.schema.cols) == 1
    # The entire row becomes one column since delimiter doesn't match
    assert dataset.data.columns == ['id,name,value']


def test_from_json_stringio(conn: DuckDBPyConnection) -> None:
    """Test JSON reading from StringIO."""
    json_content = '''{"id": 1, "name": "Alice", "active": true}
{"id": 2, "name": "Bob", "active": false}
{"id": 3, "name": "Charlie", "active": true}'''
    json_io = StringIO(json_content)
    
    source = DatasetSource.from_json(json_io, conn)
    dataset = source.to_dataset(conn)
    
    assert len(dataset) == 3
    assert dataset.data['id'].to_list() == [1, 2, 3]
    assert dataset.data['name'].to_list() == ['Alice', 'Bob', 'Charlie']
    assert dataset.data['active'].to_list() == [True, False, True]
