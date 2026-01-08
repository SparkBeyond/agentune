"""Tests for schema formatters."""

import polars as pl
import pytest
from duckdb import DuckDBPyConnection

from agentune.analyze.join.base import (
    TablesWithJoinStrategies,
    TableWithJoinStrategies,
)
from agentune.analyze.join.lookup import LookupJoinStrategy
from agentune.core import types
from agentune.core.database import DuckdbName, DuckdbTable
from agentune.core.dataset import Dataset
from agentune.core.formatter.schema import SimpleSchemaFormatter
from agentune.core.sampler.base import RandomSampler
from agentune.core.schema import Field, Schema


@pytest.fixture
def primary_dataset() -> Dataset:
    """Create a primary dataset for testing."""
    data = pl.DataFrame(
        {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
        }
    )
    return Dataset.from_polars(data)


@pytest.fixture
def secondary_table(conn: DuckDBPyConnection) -> DuckdbTable:
    """Create a secondary table in the database."""
    table_name = DuckdbName('orders', 'memory', 'main')
    schema = Schema(
        cols=(
            Field('order_id', types.int32),
            Field('customer_id', types.int32),
            Field('amount', types.float64),
        )
    )
    
    table = DuckdbTable(name=table_name, schema=schema)
    table.create(conn)
    
    # Insert sample data
    conn.execute(
        f'INSERT INTO {table_name} VALUES (101, 1, 100.5), (102, 2, 250.0), (103, 1, 75.25)'
    )
    
    return table


@pytest.fixture
def tables_with_strategies(secondary_table: DuckdbTable) -> TablesWithJoinStrategies:
    """Create tables with join strategies for testing."""
    join_strategy: LookupJoinStrategy[int] = LookupJoinStrategy(
        name='mock_join',
        table=secondary_table,
        key_col=Field('customer_id', types.int32),
        value_cols=(Field('order_id', types.int32), Field('amount', types.float64)),
    )
    table_with_strategies = TableWithJoinStrategies.from_list([join_strategy])
    return TablesWithJoinStrategies.from_list([table_with_strategies])


class TestSimpleSchemaFormatter:
    """Tests for SimpleSchemaFormatter."""

    def test_format_all_tables_basic(
        self,
        primary_dataset: Dataset,
        tables_with_strategies: TablesWithJoinStrategies,
        conn: DuckDBPyConnection,
    ) -> None:
        """Test basic formatting of all tables."""
        formatter = SimpleSchemaFormatter()
        result = formatter.format_all_tables(primary_dataset, tables_with_strategies, conn)
        
        # Check that the result contains expected sections
        assert '## Primary Table: primary_table' in result
        assert '### Schema:' in result
        assert '### Sample Data' in result
        
        # Check primary table columns are present
        assert '- id:' in result
        assert '- name:' in result
        assert '- age:' in result
        
        # Check secondary table is present
        assert '## Table: orders' in result
        assert '- order_id:' in result
        assert '- customer_id:' in result
        assert '- amount:' in result

    def test_format_all_tables_custom_primary_name(
        self,
        primary_dataset: Dataset,
        tables_with_strategies: TablesWithJoinStrategies,
        conn: DuckDBPyConnection,
    ) -> None:
        """Test formatting with custom primary table name."""
        formatter = SimpleSchemaFormatter(primary_table_name='users')
        result = formatter.format_all_tables(primary_dataset, tables_with_strategies, conn)
        
        assert '## Primary Table: users' in result
        assert '## Primary Table: primary_table' not in result

    def test_format_all_tables_custom_num_samples(
        self,
        primary_dataset: Dataset,
        tables_with_strategies: TablesWithJoinStrategies,
        conn: DuckDBPyConnection,
    ) -> None:
        """Test formatting with custom number of samples."""
        formatter = SimpleSchemaFormatter(num_samples=3)
        result = formatter.format_all_tables(primary_dataset, tables_with_strategies, conn)
        
        # Check that sample size is mentioned
        assert '### Sample Data (3 rows):' in result
        
        # Count the number of data rows in the primary table sample
        # The CSV output should have a header and 3 data rows
        primary_section = result.split('## Table: orders')[0]
        csv_lines = primary_section.split('### Sample Data (3 rows):')[1].strip().split('\n')
        # Filter out empty lines
        csv_lines = [line for line in csv_lines if line.strip()]
        # Should have header + 3 data rows
        assert len(csv_lines) == 4

    def test_format_all_tables_with_random_sampler(
        self,
        primary_dataset: Dataset,
        tables_with_strategies: TablesWithJoinStrategies,
        conn: DuckDBPyConnection,
    ) -> None:
        """Test formatting with random sampler."""
        formatter = SimpleSchemaFormatter(
            num_samples=3,
            sampler=RandomSampler(),
        )
        result = formatter.format_all_tables(
            primary_dataset, tables_with_strategies, conn, random_seed=42
        )
        
        # Should still produce valid output
        assert '## Primary Table:' in result
        assert '### Sample Data (3 rows):' in result

    def test_format_all_tables_no_secondary_tables(
        self,
        primary_dataset: Dataset,
        conn: DuckDBPyConnection,
    ) -> None:
        """Test formatting with only primary table."""
        empty_tables = TablesWithJoinStrategies.from_list([])
        formatter = SimpleSchemaFormatter()
        result = formatter.format_all_tables(primary_dataset, empty_tables, conn)
        
        # Should have primary table
        assert '## Primary Table: primary_table' in result
        assert '- id:' in result
        
        # Should not have secondary table markers
        assert result.count('## Table:') == 0

    def test_serialize_schema(self, primary_dataset: Dataset) -> None:
        """Test schema serialization."""
        formatter = SimpleSchemaFormatter()
        schema_str = formatter._serialize_schema(primary_dataset.schema)
        
        # Check that all columns are present
        assert '- id:' in schema_str
        assert '- name:' in schema_str
        assert '- age:' in schema_str
        
        # Check format
        lines = schema_str.strip().split('\n')
        assert len(lines) == 3  # Three columns
        for line in lines:
            assert line.startswith('- ')
            assert ': ' in line

    def test_format_sample_data(self, primary_dataset: Dataset) -> None:
        """Test sample data formatting."""
        formatter = SimpleSchemaFormatter()
        sample_data = formatter.sampler.sample(primary_dataset, 2)
        formatted = formatter._format_sample_data(sample_data)
        
        # Should be CSV format
        assert 'id,name,age' in formatted or 'id,' in formatted
        lines = formatted.strip().split('\n')
        # Header + 2 data rows
        assert len(lines) == 3

    def test_format_all_tables_large_dataset(
        self,
        tables_with_strategies: TablesWithJoinStrategies,
        conn: DuckDBPyConnection,
    ) -> None:
        """Test formatting with dataset larger than sample size."""
        # Create a larger dataset
        large_data = pl.DataFrame(
            {
                'id': range(100),
                'value': [f'value_{i}' for i in range(100)],
            }
        )
        large_dataset = Dataset.from_polars(large_data)
        
        formatter = SimpleSchemaFormatter(num_samples=5)
        result = formatter.format_all_tables(large_dataset, tables_with_strategies, conn)
        
        # Should only sample 5 rows
        assert '### Sample Data (5 rows):' in result
        
        # The CSV output should have limited rows
        primary_section = result.split('## Table:')[0]
        assert 'value_99' not in primary_section  # Should not contain the last row

    def test_format_all_tables_multiple_secondary_tables(
        self,
        primary_dataset: Dataset,
        conn: DuckDBPyConnection,
    ) -> None:
        """Test formatting with multiple secondary tables."""
        # Create first secondary table
        table1_name = DuckdbName('orders', 'memory', 'main')
        table1_schema = Schema(cols=(Field('order_id', types.int32),))
        table1 = DuckdbTable(name=table1_name, schema=table1_schema)
        table1.create(conn)
        conn.execute(f'INSERT INTO {table1_name} VALUES (1), (2)')
        
        # Create second secondary table
        table2_name = DuckdbName('products', 'memory', 'main')
        table2_schema = Schema(cols=(Field('product_id', types.int32),))
        table2 = DuckdbTable(name=table2_name, schema=table2_schema)
        table2.create(conn)
        conn.execute(f'INSERT INTO {table2_name} VALUES (10), (20)')
        
        # Create join strategies
        strategy1: LookupJoinStrategy[int] = LookupJoinStrategy(
            name='join1',
            table=table1,
            key_col=Field('order_id', types.int32),
            value_cols=(),
        )
        strategy2: LookupJoinStrategy[int] = LookupJoinStrategy(
            name='join2',
            table=table2,
            key_col=Field('product_id', types.int32),
            value_cols=(),
        )
        
        tables = TablesWithJoinStrategies.from_list([
            TableWithJoinStrategies.from_list([strategy1]),
            TableWithJoinStrategies.from_list([strategy2]),
        ])
        
        formatter = SimpleSchemaFormatter()
        result = formatter.format_all_tables(primary_dataset, tables, conn)
        
        # Check both secondary tables are present
        assert '## Table: orders' in result
        assert '## Table: products' in result
        assert '- order_id:' in result
        assert '- product_id:' in result

    def test_format_preserves_order(
        self,
        primary_dataset: Dataset,
        tables_with_strategies: TablesWithJoinStrategies,
        conn: DuckDBPyConnection,
    ) -> None:
        """Test that formatting preserves expected section order."""
        formatter = SimpleSchemaFormatter()
        result = formatter.format_all_tables(primary_dataset, tables_with_strategies, conn)
        
        # Find positions of key sections
        primary_pos = result.find('## Primary Table:')
        primary_schema_pos = result.find('### Schema:', primary_pos)
        primary_sample_pos = result.find('### Sample Data', primary_schema_pos)
        secondary_pos = result.find('## Table:', primary_sample_pos)
        
        # Verify order
        assert primary_pos < primary_schema_pos < primary_sample_pos < secondary_pos

    def test_format_output_example(
        self,
        primary_dataset: Dataset,
        tables_with_strategies: TablesWithJoinStrategies,
        conn: DuckDBPyConnection,
    ) -> None:
        """Test to display the actual formatted output for visual inspection."""
        formatter = SimpleSchemaFormatter(num_samples=3)
        result = formatter.format_all_tables(primary_dataset, tables_with_strategies, conn)
        
        # Print the actual output for inspection
        print('\n' + '=' * 80)
        print('FORMATTED OUTPUT:')
        print('=' * 80)
        print(result)
        print('=' * 80)
        
        # Basic assertions to ensure test passes
        assert len(result) > 0
        assert '## Primary Table:' in result
