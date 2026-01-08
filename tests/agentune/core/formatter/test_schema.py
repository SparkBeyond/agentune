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
