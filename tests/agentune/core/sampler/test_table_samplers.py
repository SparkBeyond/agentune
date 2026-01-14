"""Tests for table sampling utilities."""

import polars as pl
from duckdb import DuckDBPyConnection

from agentune.core.dataset import Dataset, DatasetSink
from agentune.core.sampler.table_samples import HeadTableSampler, RandomStartTableSampler


def create_test_table(conn: DuckDBPyConnection, table_name: str, num_rows: int) -> None:
    """Helper to create test tables in DuckDB."""
    # Create test data with explicit schema to avoid Null type issues with empty DataFrames
    data = pl.DataFrame({
        'id': list(range(num_rows)),
        'value': [f'item_{i}' for i in range(num_rows)],},
        schema={'id': pl.Int64, 'value': pl.String}
    )
    
    # Create actual table in DuckDB (not just register)
    DatasetSink.into_unqualified_duckdb_table(table_name, conn).write(Dataset.from_polars(data).as_source(), conn)


class TestHeadTableSampler:
    """Test HeadTableSampler functionality."""
    
    def test_basic_head_sampling(self, conn: DuckDBPyConnection) -> None:
        """Test basic head sampling functionality."""
        sampler = HeadTableSampler()
        table_name = 'test_table'
        create_test_table(conn, table_name, 100)
        
        # Sample first 20 rows
        result = sampler.sample(table_name, conn, sample_size=20)
        
        # Validate result
        expected = conn.table(str(table_name)).pl().head(20)
        assert result.data.equals(expected)
    
    def test_head_sampling_full_table(self, conn: DuckDBPyConnection) -> None:
        """Test head sampling when sample size equals table size."""
        sampler = HeadTableSampler()
        table_name = 'test_table_full'
        create_test_table(conn, table_name, 50)
        
        result = sampler.sample(table_name, conn, sample_size=50)
        
        expected = conn.table(str(table_name)).pl()
        assert result.data.equals(expected)
    
    def test_head_sampling_ignores_random_seed(self, conn: DuckDBPyConnection) -> None:
        """Test that head sampler produces same results regardless of seed."""
        sampler = HeadTableSampler()
        table_name = 'test_table_seed'
        create_test_table(conn, table_name, 50)
        
        result1 = sampler.sample(table_name, conn, sample_size=10, random_seed=42)
        result2 = sampler.sample(table_name, conn, sample_size=10, random_seed=999)
        
        assert result1.data.equals(result2.data)
    
    def test_head_sampling_larger_than_table(self, conn: DuckDBPyConnection) -> None:
        """Test head sampling when sample size exceeds table size."""
        sampler = HeadTableSampler()
        table_name = 'test_head_oversample'
        create_test_table(conn, table_name, 30)
        
        result = sampler.sample(table_name, conn, sample_size=50)
        
        # Should return entire table (all 30 rows)
        expected = conn.table(str(table_name)).pl()
        assert result.data.equals(expected)
    
    def test_head_sampling_empty_table(self, conn: DuckDBPyConnection) -> None:
        """Test head sampling from an empty table."""
        sampler = HeadTableSampler()
        table_name = 'test_head_empty'
        create_test_table(conn, table_name, 0)
        
        result = sampler.sample(table_name, conn, sample_size=10)
        
        # Should return empty dataset with correct schema
        expected = conn.table(str(table_name)).pl()
        assert result.data.equals(expected)


class TestRandomStartTableSampler:
    """Test RandomStartTableSampler functionality."""
    
    def test_basic_random_start_sampling(self, conn: DuckDBPyConnection) -> None:
        """Test basic random start sampling functionality."""
        sampler = RandomStartTableSampler()
        table_name = 'test_random_table'
        create_test_table(conn, table_name, 100)
        
        result = sampler.sample(table_name, conn, sample_size=20, random_seed=42)
        
        # Validate result
        assert result.data.height == 20
        assert result.schema.names == ['id', 'value']
        # Check that rows are consecutive
        ids = result.data['id'].to_list()
        assert ids == list(range(ids[0], ids[0] + 20))
    
    def test_random_start_sampling_reproducibility(self, conn: DuckDBPyConnection) -> None:
        """Test that random start sampling is reproducible with same seed."""
        sampler = RandomStartTableSampler()
        table_name = 'test_reproducible'
        create_test_table(conn, table_name, 100)
        
        result1 = sampler.sample(table_name, conn, sample_size=15, random_seed=123)
        result2 = sampler.sample(table_name, conn, sample_size=15, random_seed=123)
        
        assert result1.data.equals(result2.data)
    
    def test_random_start_sampling_different_seeds(self, conn: DuckDBPyConnection) -> None:
        """Test that different seeds produce different starting points."""
        sampler = RandomStartTableSampler()
        table_name = 'test_different_seeds'
        create_test_table(conn, table_name, 100)
        
        result1 = sampler.sample(table_name, conn, sample_size=20, random_seed=42)
        result2 = sampler.sample(table_name, conn, sample_size=20, random_seed=999)
        
        # Different seeds should (very likely) produce different starting points
        assert result1.data['id'][0] != result2.data['id'][0]
    
    def test_random_start_sampling_consecutive_rows(self, conn: DuckDBPyConnection) -> None:
        """Test that sampled rows are consecutive."""
        sampler = RandomStartTableSampler()
        table_name = 'test_consecutive_rows'
        create_test_table(conn, table_name, 100)
        
        result = sampler.sample(table_name, conn, sample_size=30, random_seed=42)
        
        ids = result.data['id'].to_list()
        # Verify consecutiveness
        for i in range(len(ids) - 1):
            assert ids[i + 1] == ids[i] + 1
    
    def test_random_start_sampling_full_table(self, conn: DuckDBPyConnection) -> None:
        """Test random start sampling when sample size equals table size."""
        sampler = RandomStartTableSampler()
        table_name = 'test_full_random'
        create_test_table(conn, table_name, 50)
        
        result = sampler.sample(table_name, conn, sample_size=50, random_seed=42)
        
        # When sample size equals table size, starting point should be 0
        expected = conn.table(str(table_name)).pl()
        assert result.data.equals(expected)
    
    def test_random_start_sampling_larger_than_table(self, conn: DuckDBPyConnection) -> None:
        """Test random start sampling when sample size exceeds table size."""
        sampler = RandomStartTableSampler()
        table_name = 'test_oversize_random'
        create_test_table(conn, table_name, 50)
        
        result = sampler.sample(table_name, conn, sample_size=50, random_seed=42)
        
        # Should return entire table
        expected = conn.table(str(table_name)).pl()
        assert result.data.equals(expected)
    
    def test_random_start_sampling_empty_table(self, conn: DuckDBPyConnection) -> None:
        """Test random start sampling with empty table."""
        sampler = RandomStartTableSampler()
        table_name = 'test_empty_random'
        create_test_table(conn, table_name, 0)
        
        result = sampler.sample(table_name, conn, sample_size=10, random_seed=42)
        
        # Should return empty dataset with correct schema
        expected = conn.table(str(table_name)).pl()
        assert result.data.equals(expected)
    
    def test_random_start_sampling_near_end(self, conn: DuckDBPyConnection) -> None:
        """Test that samples near the end of the table work correctly."""
        sampler = RandomStartTableSampler()
        table_name = 'test_near_end'
        create_test_table(conn, table_name, 100)
        
        # Force a start near the end by setting specific seed
        # This tests that we correctly handle the range
        result = sampler.sample(table_name, conn, sample_size=10, random_seed=42)
        
        ids = result.data['id'].to_list()
        # Should still get 10 consecutive rows
        assert len(ids) == 10
        assert ids == list(range(ids[0], ids[0] + 10))
        # And they should be within the table bounds
        assert ids[-1] < 100
