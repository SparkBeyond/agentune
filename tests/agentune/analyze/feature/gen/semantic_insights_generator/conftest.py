"""Fixtures for semantic insights generator tests.

TODO: Temporary placeholder. Real test data will be added in a separate task.
Should use tabular (non-conversational) data, not the conversational data
used by insightful_text_generator.
"""

import pytest
from duckdb import DuckDBPyConnection

from agentune.analyze.feature.problem import ClassificationProblem
from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.core.dataset import Dataset
from agentune.core.llm import LLMContext
from agentune.core.sercontext import LLMWithSpec


@pytest.fixture
def test_dataset_with_strategy(conn: DuckDBPyConnection) -> tuple[Dataset, str, TablesWithJoinStrategies]:
    """Temporary placeholder. Real test data will be added in a separate task."""
    raise NotImplementedError('TODO: Define test data for semantic insights generator')


@pytest.fixture
async def real_llm_with_spec(llm_context_nocache: LLMContext) -> LLMWithSpec:
    """Temporary placeholder. Real test data will be added in a separate task."""
    raise NotImplementedError('TODO: Define LLM fixture for semantic insights generator')


@pytest.fixture
def problem(test_dataset_with_strategy: tuple[Dataset, str, TablesWithJoinStrategies]) -> ClassificationProblem:
    """Temporary placeholder. Real test data will be added in a separate task."""
    raise NotImplementedError('TODO: Define problem fixture for semantic insights generator')
