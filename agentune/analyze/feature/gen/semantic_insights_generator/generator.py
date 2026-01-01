"""SemanticInsightsGenerator - SQL-based feature generation using LLMs."""

from __future__ import annotations

from collections.abc import AsyncIterator

from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.feature.gen.base import FeatureGenerator, GeneratedFeature
from agentune.analyze.feature.gen.semantic_insights_generator.basic_generator import (
    BasicFeatureGenerator,
)
from agentune.analyze.feature.problem import Problem
from agentune.analyze.feature.validate.law_and_order import LawAndOrderValidator
from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.core.dataset import Dataset
from agentune.core.sercontext import LLMWithSpec


@frozen
class SemanticInsightsGenerator(FeatureGenerator):
    """Feature generator that creates SQL-based features using LLMs.

    This generator:
    - Uses a reasoning model to generate SQL feature specifications
    - Validates features using LawAndOrderValidator
    - Repairs failed features using a fast repair model
    - Integrates with the official validation framework

    Attributes:
        generation_model: LLM for generating feature SQL (e.g., claude-opus-4-5)
        repair_model: LLM for repairing validation errors (e.g., claude-haiku-4)
        seed: Random seed for reproducible generation (None = non-deterministic)
    """

    generation_model: LLMWithSpec
    repair_model: LLMWithSpec
    seed: int | None = None

    async def agenerate(
        self,
        feature_search: Dataset,
        problem: Problem,
        join_strategies: TablesWithJoinStrategies,
        conn: DuckDBPyConnection,
    ) -> AsyncIterator[GeneratedFeature]:
        """Generate SQL-based features for the given problem.

        Args:
            feature_search: Input dataset for feature generation context
            problem: Problem specification (target column, problem type)
            join_strategies: Available join strategies for secondary tables
            conn: DuckDB connection for SQL execution

        Yields:
            GeneratedFeature instances with has_good_defaults=False
        """
        basic_gen = BasicFeatureGenerator(
            generation_model=self.generation_model,
            repair_model=self.repair_model,
            seed=self.seed,
            validators=(LawAndOrderValidator(),),
        )

        async for gen_feature in basic_gen.agenerate(
            dataset=feature_search,
            problem=problem,
            join_strategies=join_strategies,
            conn=conn,
        ):
            yield gen_feature
