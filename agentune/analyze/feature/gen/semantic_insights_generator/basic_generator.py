"""Basic feature generator - executes one generation cycle."""

from __future__ import annotations

from collections.abc import AsyncIterator

from attrs import define, field
from duckdb import DuckDBPyConnection

from agentune.analyze.feature.gen.base import GeneratedFeature
from agentune.analyze.feature.gen.semantic_insights_generator.corrector import (
    LLMSqlFeatureCorrector,
)
from agentune.analyze.feature.gen.semantic_insights_generator.llm.schema import (
    GeneratedFeatureSpec,
)
from agentune.analyze.feature.problem import Problem
from agentune.analyze.feature.sql.create import feature_from_query
from agentune.analyze.feature.sql.validator_loop import ValidateAndRetryParams, validate_and_retry
from agentune.analyze.feature.validate.base import FeatureValidator
from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.core.dataset import Dataset
from agentune.core.sercontext import LLMWithSpec

# Default validation retry budgets
_DEFAULT_MAX_GLOBAL_RETRIES = 5
_DEFAULT_MAX_LOCAL_RETRIES = 3


@define
class BasicFeatureGenerator:
    """Executes one complete generation cycle for SQL-based features.

    This component handles:
    - Calling LLM to generate feature specifications
    - Validating each feature using validate_and_retry()
    - Yielding validated features as they're ready (streaming)
    """

    generation_model: LLMWithSpec
    repair_model: LLMWithSpec
    seed: int | None = None
    validators: tuple[FeatureValidator, ...] = field(factory=tuple)
    max_global_retries: int = _DEFAULT_MAX_GLOBAL_RETRIES
    max_local_retries: int = _DEFAULT_MAX_LOCAL_RETRIES

    async def agenerate(
        self,
        dataset: Dataset,
        problem: Problem,  # noqa: ARG002 - Used in TODO: LLM prompt generation
        join_strategies: TablesWithJoinStrategies,
        conn: DuckDBPyConnection,
    ) -> AsyncIterator[GeneratedFeature]:
        """Generate features using a single LLM call cycle.

        Args:
            dataset: Input dataset for feature generation context
            problem: Problem specification (target column, problem type)
            join_strategies: Available join strategies for secondary tables
            conn: DuckDB connection for SQL execution

        Yields:
            GeneratedFeature instances as they pass validation
        """
        # TODO: Steps 1-3 - LLM generation (implement later)
        # 1. Sample data for LLM context using self.seed
        # 2. Build prompt with problem description, schema, samples (needs `problem`)
        # 3. Call generation_model to get feature specifications

        # Step 4: Validate each feature using validate_and_retry
        feature_specs: list[GeneratedFeatureSpec] = []  # Placeholder empty list

        # Extract secondary tables from join_strategies
        secondary_tables = [tws.table for tws in join_strategies]

        # TODO: Sample data for validation (use self.seed)
        # For now, use full dataset
        sampled_data = dataset

        # Validate and yield features
        for spec in feature_specs:
            # Create corrector for this feature
            corrector = LLMSqlFeatureCorrector(
                repair_model=self.repair_model,
            )

            # Validate feature with retry loop
            feature = await validate_and_retry(
                ValidateAndRetryParams(
                    feature_ctor=feature_from_query,
                    conn=conn,
                    sql_query=spec.sql_query,
                    params=dataset.schema,
                    secondary_tables=secondary_tables,
                    input=sampled_data,
                    max_global_retries=self.max_global_retries,
                    max_local_retries=self.max_local_retries,
                    corrector=corrector,
                    validators=self.validators,
                    name=spec.name,
                    description=spec.description,
                    technical_description=spec.sql_query,
                )
            )

            # Yield if validation succeeded
            if feature:
                yield GeneratedFeature(feature=feature, has_good_defaults=False)
