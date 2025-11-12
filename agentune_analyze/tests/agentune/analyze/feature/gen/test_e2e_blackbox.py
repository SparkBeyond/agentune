"""Black-box end-to-end test for feature generation using only public API.

This test validates the complete feature generation pipeline by using only the
public API (agenerate()), without accessing internal implementation details.
It focuses on validating the final output rather than intermediate steps.
"""

import logging
import math
from pathlib import Path

import httpx
import polars as pl
import pytest
from duckdb import DuckDBPyConnection

import agentune.analyze.core.types as dtypes
from agentune.analyze.core import duckdbio
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.llm import LLMContext, LLMSpec
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.feature.base import CategoricalFeature
from agentune.analyze.feature.gen.insightful_text_generator.insightful_text_generator import (
    ConversationQueryFeatureGenerator,
)
from agentune.analyze.feature.problem import ClassificationProblem, ProblemDescription
from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.analyze.join.conversation import ConversationJoinStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def test_dataset_with_strategy(test_data_conversations: dict[str, Path], conn: DuckDBPyConnection) -> tuple[Dataset, str, TablesWithJoinStrategies]:
    """Load and prepare test data for ConversationQueryGenerator."""
    # Load CSV files
    main_df = pl.read_csv(test_data_conversations['main_csv'])
    conversations_df = pl.read_csv(test_data_conversations['conversations_csv'])
    
    # Create schemas
    main_schema = Schema((
        Field(name='id', dtype=dtypes.int32),
        Field(name='outcome', dtype=dtypes.EnumDtype(*main_df['outcome'].unique().to_list())),
        Field(name='outcome_description', dtype=dtypes.string),
    ))
    # convert columns to appropriate types
    for field in main_schema.cols:
        main_df = main_df.with_columns(pl.col(field.name).cast(field.dtype.polars_type))

    # Create secondary table schema
    secondary_schema = Schema((
        Field(name='id', dtype=dtypes.int32),
        Field(name='timestamp', dtype=dtypes.timestamp),
        Field(name='role', dtype=dtypes.string),
        Field(name='content', dtype=dtypes.string),
    ))
    # convert columns to appropriate types
    conversations_df = conversations_df.with_columns(
        pl.col('timestamp').str.to_datetime('%Y-%m-%dT%H:%M:%SZ')
    )
    for field in secondary_schema.cols:
        conversations_df = conversations_df.with_columns(pl.col(field.name).cast(field.dtype.polars_type))

    # Create datasets
    main_dataset = Dataset(schema=main_schema, data=main_df)
    secondary_dataset = Dataset(schema=secondary_schema, data=conversations_df)

    # Ingest tables
    duckdbio.ingest(conn, 'main', main_dataset.as_source())
    context_table = duckdbio.ingest(conn, 'conversations', secondary_dataset.as_source())

    conversation_strategy = ConversationJoinStrategy[int].on_table(
        'conversations',
        context_table.table,
        'id',           # main_table_id_column
        'id',           # id_column
        'timestamp',    # timestamp_column
        'role',         # role_column
        'content'       # content_column
    )

    # Create index
    conversation_strategy.index.create(conn, context_table.table.name, if_not_exists=True)

    strategies = TablesWithJoinStrategies.group([conversation_strategy])

    return main_dataset, 'outcome', strategies


@pytest.fixture
async def real_llm_with_spec(httpx_async_client: httpx.AsyncClient) -> LLMWithSpec:
    """Create a real LLM for end-to-end testing."""
    llm_context = LLMContext(httpx_async_client)
    llm_spec = LLMSpec('openai', 'gpt-4o-mini')  # Use a smaller, faster model for testing
    llm_with_spec = LLMWithSpec(
        llm=llm_context.from_spec(llm_spec),
        spec=llm_spec
    )
    return llm_with_spec


@pytest.fixture
def problem(test_dataset_with_strategy: tuple[Dataset, str, TablesWithJoinStrategies]) -> ClassificationProblem:
    """Create a Problem fixture for testing."""
    main_dataset, target_col, _ = test_dataset_with_strategy
    
    # Get the target field from the dataset
    target_field = main_dataset.schema[target_col]
    
    # Get unique classes from the target column
    unique_values = main_dataset.data.get_column(target_col).unique().to_list()
    classes = tuple(sorted(unique_values))
    
    # Create problem description
    problem_description = ProblemDescription(
        target_column=target_col,
        problem_type='classification',
        target_desired_outcome='resolved',
        name='Customer Support Resolution Prediction',
        description='Predict whether customer support conversations will be resolved successfully',
        target_description='Whether the conversation resulted in a resolved outcome',
        business_domain='customer support',
        comments='Generated for testing the conversation feature generator'
    )
    
    # Create classification problem
    return ClassificationProblem(
        problem_description=problem_description,
        target_column=target_field,
        classes=classes
    )


@pytest.mark.integration
async def test_blackbox_feature_generation_with_real_llm(test_dataset_with_strategy: tuple[Dataset, str, TablesWithJoinStrategies],
                                                         conn: DuckDBPyConnection,
                                                         real_llm_with_spec: LLMWithSpec,
                                                         problem: ClassificationProblem) -> None:
    """Test the complete feature generation pipeline using only the public API.
    
    This is a black-box test that validates the end-to-end workflow without
    inspecting intermediate steps like query generation or enrichment.
    """
    main_dataset, target_col, strategies = test_dataset_with_strategy
    random_seed = 42

    feature_generator = ConversationQueryFeatureGenerator(
        query_generator_model=real_llm_with_spec,
        max_samples_for_generation=10,
        num_samples_for_enrichment=5,
        num_features_per_round=5,
        num_actionable_rounds=2,
        num_creative_features=2,
        query_enrich_model=real_llm_with_spec,
        random_seed=random_seed
    )

    # Use the public API - agenerate() returns fully-formed features
    generated_features = []
    async for gen_feature in feature_generator.agenerate(main_dataset, problem, strategies, conn):
        generated_features.append(gen_feature)
    
    # Validate that features were generated
    assert len(generated_features) > 0, 'Should generate at least one feature'
    logger.info(f'Generated {len(generated_features)} features')
    
    # Validate each generated feature has required properties
    for gen_feature in generated_features:
        feature = gen_feature.feature
        assert feature.name is not None
        assert feature.description is not None
        assert feature.dtype is not None
        logger.info(f'Feature: {feature.name} (type: {feature.dtype})')
    
    # Test feature evaluation on first row
    feature = generated_features[0].feature
    strict_df = pl.DataFrame([main_dataset.data.get_column(col.name) for col in feature.params.cols])
    first_row_args = strict_df.row(0, named=False)
    result = await feature.acompute(first_row_args, conn)
    
    # Validate result
    if result is None:
        logger.info(f'Feature {feature.name} returned None (missing value)')
    else:
        # Check that the type of the result matches the expected Python type from DuckDB
        expected_python_type = dtypes.python_type_from_polars(feature.dtype)
        assert isinstance(result, expected_python_type), f'Feature {feature.name} returned {type(result)} but expected {expected_python_type}'
        
        # Additional type-specific validation
        if feature.dtype.is_numeric():
            # For numeric types, check that the result is not NaN or infinite
            assert isinstance(result, float | int), f'Feature {feature.name} returned non-numeric type {type(result)}'
            assert not math.isnan(float(result)), f'Feature {feature.name} returned NaN'
            assert not math.isinf(float(result)), f'Feature {feature.name} returned infinite value'
        elif isinstance(feature, CategoricalFeature):
            # For categorical/enum types, check that the result is one of the valid categories
            valid_categories = [*list(feature.categories), CategoricalFeature.other_category]
            assert result in valid_categories, (
                f'Feature {feature.name} returned "{result}" which is not in valid categories: '
                f'{feature.categories}'
            )
        
        logger.info(f'Feature {feature.name} evaluation successful: {result} (type: {type(result).__name__})')
    
    logger.info('Black-box end-to-end test completed successfully')
