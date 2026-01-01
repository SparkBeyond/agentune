"""E2E tests for SemanticInsightsGenerator."""

import pytest
from duckdb import DuckDBPyConnection

from agentune.analyze.feature.gen.semantic_insights_generator import SemanticInsightsGenerator
from agentune.analyze.feature.problem import ClassificationProblem
from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.core.dataset import Dataset
from agentune.core.sercontext import LLMWithSpec


@pytest.mark.integration
async def test_semantic_insights_generator(
    test_dataset_with_strategy: tuple[Dataset, str, TablesWithJoinStrategies],
    conn: DuckDBPyConnection,
    real_llm_with_spec: LLMWithSpec,
    problem: ClassificationProblem,
) -> None:
    """Test that SemanticInsightsGenerator can be instantiated and API works.

    This is a minimal test validating the generator structure. It expects
    empty results until LLM generation is implemented in BasicFeatureGenerator.

    NOTE: When LLM generation is implemented in BasicFeatureGenerator:
    1. Remove the "assert len(features) == 0" check
    2. Add assertions for generated features (structure, computation, validation)
    3. Consider adding more comprehensive tests following test_e2e.py patterns
    """
    # Unpack test data
    main_dataset, target_col, strategies = test_dataset_with_strategy

    # Create generator with both models
    generator = SemanticInsightsGenerator(
        generation_model=real_llm_with_spec,
        repair_model=real_llm_with_spec,
        seed=42,
    )

    # Call agenerate - should not yield features yet (stub implementation)
    features = []
    async for gen_feature in generator.agenerate(
        feature_search=main_dataset,
        problem=problem,
        join_strategies=strategies,
        conn=conn,
    ):
        features.append(gen_feature)

    # Expect empty list until LLM generation is implemented
    assert len(features) == 0, (
        'Expected no features from stub implementation. '
        'If this fails, LLM generation has been implemented - update test!'
    )
