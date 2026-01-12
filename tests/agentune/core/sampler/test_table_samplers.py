"""Tests for table sampling utilities."""

import polars as pl
from duckdb import DuckDBPyConnection

from agentune.analyze.join.base import TableWithJoinStrategies
from agentune.core.database import DuckdbName, DuckdbTable
from agentune.core.sampler.table_samples import HeadTableSampler, RandomStartTableSampler
from agentune.core.schema import Field, Schema
from agentune.core.types import int32, string


def create_test_table(conn: DuckDBPyConnection, table_name: str, num_rows: int) -> TableWithJoinStrategies:
    """Helper to create test tables in DuckDB."""
    # Create test data
    data = pl.DataFrame({
        'id': list(range(num_rows)),
        'value': [f'item_{i}' for i in range(num_rows)],
    })
    
    # Create actual table in DuckDB (not just register)
    qualified_name = DuckdbName.qualify(table_name, conn)
    schema = Schema((Field('id', int32), Field('value', string)))
    duckdb_table = DuckdbTable(name=qualified_name, schema=schema)
    
    # Create the table
    duckdb_table.create(conn, if_not_exists=True)
    
    # Insert data
    if num_rows > 0:
        conn.register('__temp_data', data)
        conn.execute(f'INSERT INTO {qualified_name} SELECT * FROM __temp_data')
        conn.unregister('__temp_data')
    
    # Return TableWithJoinStrategies
    return TableWithJoinStrategies(table=duckdb_table, join_strategies={})


class TestHeadTableSampler:
    """Test HeadTableSampler functionality."""
    
    def test_basic_head_sampling(self, conn: DuckDBPyConnection) -> None:
        """Test basic head sampling functionality."""
        sampler = HeadTableSampler()
        table = create_test_table(conn, 'test_table', 100)
        
        # Sample first 20 rows
        result = sampler.sample(table, conn, sample_size=20)
        
        # Validate result
        assert result.data.height == 20
        assert result.data['id'].to_list() == list(range(20))
        assert result.schema.names == ['id', 'value']
    
    def test_head_sampling_full_table(self, conn: DuckDBPyConnection) -> None:
        """Test head sampling when sample size equals table size."""
        sampler = HeadTableSampler()
        table = create_test_table(conn, 'test_table_full', 50)
        
        result = sampler.sample(table, conn, sample_size=50)
        
        assert result.data.height == 50
        assert result.data['id'].to_list() == list(range(50))
    
    def test_head_sampling_ignores_random_seed(self, conn: DuckDBPyConnection) -> None:
        """Test that head sampler produces same results regardless of seed."""
        sampler = HeadTableSampler()
        table = create_test_table(conn, 'test_table_seed', 50)
        
        result1 = sampler.sample(table, conn, sample_size=10, random_seed=42)
        result2 = sampler.sample(table, conn, sample_size=10, random_seed=999)
        
        assert result1.data.equals(result2.data)
    
    def test_head_sampling_larger_than_table(self, conn: DuckDBPyConnection) -> None:
        """Test head sampling when sample size exceeds table size."""
        sampler = HeadTableSampler()
        table = create_test_table(conn, 'test_head_oversample', 30)
        
        result = sampler.sample(table, conn, sample_size=50)
        
        # Should return entire table (all 30 rows)
        assert result.data.height == 30
        assert result.data['id'].to_list() == list(range(30))
    
    def test_head_sampling_empty_table(self, conn: DuckDBPyConnection) -> None:
        """Test head sampling from an empty table."""
        sampler = HeadTableSampler()
        table = create_test_table(conn, 'test_head_empty', 0)
        
        result = sampler.sample(table, conn, sample_size=10)
        
        # Should return empty dataset
        assert result.data.height == 0
        # But schema should still be correct
        assert result.schema.names == ['id', 'value']


class TestRandomStartTableSampler:
    """Test RandomStartTableSampler functionality."""
    
    def test_basic_random_start_sampling(self, conn: DuckDBPyConnection) -> None:
        """Test basic random start sampling functionality."""
        sampler = RandomStartTableSampler()
        table = create_test_table(conn, 'test_random_table', 100)
        
        result = sampler.sample(table, conn, sample_size=20, random_seed=42)
        
        # Validate result
        assert result.data.height == 20
        assert result.schema.names == ['id', 'value']
        # Check that rows are consecutive
        ids = result.data['id'].to_list()
        assert ids == list(range(ids[0], ids[0] + 20))
    
    def test_random_start_sampling_reproducibility(self, conn: DuckDBPyConnection) -> None:
        """Test that random start sampling is reproducible with same seed."""
        sampler = RandomStartTableSampler()
        table = create_test_table(conn, 'test_reproducible', 100)
        
        result1 = sampler.sample(table, conn, sample_size=15, random_seed=123)
        result2 = sampler.sample(table, conn, sample_size=15, random_seed=123)
        
        assert result1.data.equals(result2.data)
    
    def test_random_start_sampling_different_seeds(self, conn: DuckDBPyConnection) -> None:
        """Test that different seeds produce different starting points."""
        sampler = RandomStartTableSampler()
        table = create_test_table(conn, 'test_different_seeds', 100)
        
        result1 = sampler.sample(table, conn, sample_size=20, random_seed=42)
        result2 = sampler.sample(table, conn, sample_size=20, random_seed=999)
        
        # Different seeds should (very likely) produce different starting points
        ids1 = result1.data['id'].to_list()
        ids2 = result2.data['id'].to_list()
        assert ids1[0] != ids2[0] or ids1 != ids2
    
    def test_random_start_sampling_consecutive_rows(self, conn: DuckDBPyConnection) -> None:
        """Test that sampled rows are consecutive."""
        sampler = RandomStartTableSampler()
        table = create_test_table(conn, 'test_consecutive', 200)
        
        result = sampler.sample(table, conn, sample_size=30, random_seed=42)
        
        ids = result.data['id'].to_list()
        # Verify consecutiveness
        for i in range(len(ids) - 1):
            assert ids[i + 1] == ids[i] + 1
    
    def test_random_start_sampling_full_table(self, conn: DuckDBPyConnection) -> None:
        """Test random start sampling when sample size equals table size."""
        sampler = RandomStartTableSampler()
        table = create_test_table(conn, 'test_full_random', 50)
        
        result = sampler.sample(table, conn, sample_size=50, random_seed=42)
        
        # When sample size equals table size, starting point should be 0
        assert result.data.height == 50
        assert result.data['id'].to_list() == list(range(50))
    
    def test_random_start_sampling_larger_than_table(self, conn: DuckDBPyConnection) -> None:
        """Test random start sampling when sample size exceeds table size."""
        sampler = RandomStartTableSampler()
        table = create_test_table(conn, 'test_oversample', 30)
        
        result = sampler.sample(table, conn, sample_size=50, random_seed=42)
        
        # Should return entire table
        assert result.data.height == 30
        assert result.data['id'].to_list() == list(range(30))
        # Schema should be correct
        assert result.schema.names == ['id', 'value']
    
    def test_random_start_sampling_empty_table(self, conn: DuckDBPyConnection) -> None:
        """Test random start sampling with empty table."""
        sampler = RandomStartTableSampler()
        table = create_test_table(conn, 'test_empty', 0)
        
        result = sampler.sample(table, conn, sample_size=10, random_seed=42)
        
        # Should return empty dataset
        assert result.data.height == 0
        # But schema should still be correct
        assert result.schema.names == ['id', 'value']
    
    def test_random_start_sampling_near_end(self, conn: DuckDBPyConnection) -> None:
        """Test that samples near the end of the table work correctly."""
        sampler = RandomStartTableSampler()
        table = create_test_table(conn, 'test_near_end', 100)
        
        # Force a start near the end by setting specific seed
        # This tests that we correctly handle the range
        result = sampler.sample(table, conn, sample_size=10, random_seed=42)
        
        ids = result.data['id'].to_list()
        # Should still get 10 consecutive rows
        assert len(ids) == 10
        assert ids == list(range(ids[0], ids[0] + 10))
        # And they should be within the table bounds
        assert ids[-1] < 100
