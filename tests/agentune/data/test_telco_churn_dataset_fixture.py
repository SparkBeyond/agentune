"""Test that the sampled_telco_churn fixture works correctly."""

import pytest
from tests.agentune.conftest import TestStructuredDataset

from agentune.api.base import RunContext


@pytest.mark.asyncio
async def test_golden_dataset_fixture_loads_data(sampled_telco_churn: TestStructuredDataset, ctx: RunContext) -> None:
    """Verify sampled_telco_churn fixture loads all tables and creates valid problem."""
    dataset = sampled_telco_churn
    
    # Check table names are set
    assert dataset.train_table == 'train'
    assert dataset.test_table == 'test'
    assert len(dataset.auxiliary_tables) == 3
    assert 'billing_history_table' in dataset.auxiliary_tables
    assert 'top_up_activation_history_table' in dataset.auxiliary_tables
    assert 'customer_feedback_table' in dataset.auxiliary_tables
    
    # Check problem is configured correctly
    assert dataset.problem.target_column.name == 'churn_status'
    assert dataset.problem.date_column.name == 'reference_date'
    assert dataset.problem.target_kind == 'classification'
    assert dataset.problem.classes == (0, 1)
    
    # Verify tables actually exist in DuckDB and have data
    with ctx._ddb_manager.cursor() as conn:
        train_count = conn.execute(f'SELECT COUNT(*) FROM {dataset.train_table}').fetchone()[0]
        test_count = conn.execute(f'SELECT COUNT(*) FROM {dataset.test_table}').fetchone()[0]
        billing_count = conn.execute('SELECT COUNT(*) FROM billing_history_table').fetchone()[0]
        
        # Verify tables have data
        assert train_count > 0, 'Train table is empty'
        assert test_count > 0, 'Test table is empty'
        assert billing_count > 0, 'Billing history table is empty'
        
        # Verify target column exists and has binary values
        target_values = conn.execute(f'SELECT DISTINCT {dataset.problem.target_column.name} FROM {dataset.train_table} ORDER BY 1').fetchall()
        assert len(target_values) == 2, f'Expected 2 distinct target values, got {len(target_values)}'
        assert target_values[0][0] == 0
        assert target_values[1][0] == 1
