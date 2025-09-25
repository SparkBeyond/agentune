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
from agentune.analyze.feature.gen.insightful_text_generator.features import create_feature
from agentune.analyze.feature.gen.insightful_text_generator.insightful_text_generator import (
    ConversationQueryFeatureGenerator,
)
from agentune.analyze.feature.gen.insightful_text_generator.schema import Query
from agentune.analyze.feature.problem import ClassificationProblem, ProblemDescription
from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.analyze.join.conversation import ConversationJoinStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def test_data_paths() -> dict[str, Path]:
    """Get the test data directory."""
    test_data_dir = Path(__file__).parent.parent.parent / 'data' / 'conversations'
    paths_dict = {
        'main_csv': test_data_dir / 'example_main.csv',
        'conversations_csv': test_data_dir / 'example_conversations_secondary.csv'
    }
    return paths_dict


@pytest.fixture
def test_dataset_with_strategy(test_data_paths: dict[str, Path], conn: DuckDBPyConnection) -> tuple[Dataset, str, TablesWithJoinStrategies]:
    """Load and prepare test data for ConversationQueryGenerator."""
    # Load CSV files
    main_df = pl.read_csv(test_data_paths['main_csv'])
    conversations_df = pl.read_csv(test_data_paths['conversations_csv'])
    
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
async def test_end_to_end_pipeline_with_real_llm(test_dataset_with_strategy: tuple[Dataset, str, TablesWithJoinStrategies],
                                                 conn: DuckDBPyConnection,
                                                 real_llm_with_spec: LLMWithSpec,
                                                 problem: ClassificationProblem) -> None:
    """Test the complete end-to-end pipeline with real LLM."""
    main_dataset, target_col, strategies = test_dataset_with_strategy
    random_seed = 42  # Use a fixed seed for reproducibility

    feature_generator: ConversationQueryFeatureGenerator = ConversationQueryFeatureGenerator(
        query_generator_model=real_llm_with_spec,
        num_features_to_generate=5,
        num_samples_for_generation=10,
        num_samples_for_enrichment=5,
        query_enrich_model=real_llm_with_spec,
        random_seed=random_seed
    )

    # imitate the feature search
    conversation_strategies = feature_generator.find_conversation_strategies(strategies)

    for conversation_strategy in conversation_strategies:
        # 1. Create a query generator for the conversation strategy
        query_generator = feature_generator.create_query_generator(conversation_strategy, problem)

        # 2. Generate queries from the conversation data
        queries = await query_generator.agenerate_queries(main_dataset, problem, conn, random_seed=feature_generator.random_seed)

        # Validate result
        assert isinstance(queries, list)
        assert len(queries) > 0
        assert all(isinstance(q, Query) for q in queries)

        # Validate individual queries
        for query in queries:
            assert query.name is not None
            assert query.query_text is not None
            assert query.return_type is not None
            
            # Check query format
            assert isinstance(query.name, str)
            assert isinstance(query.query_text, str)
            assert len(query.name) > 0
            assert len(query.query_text) > 0

            # log the generated queries for verification
            logger.info(f'Generated queries: {query.name} - {query.query_text}')

        # 3. Test enrichment part
        sampler = feature_generator._get_sampler(problem)
        sampled_data = sampler.sample(main_dataset, feature_generator.num_samples_for_enrichment, feature_generator.random_seed)
        enrichment_formatter = feature_generator._get_formatter(conversation_strategy, problem, include_target=False)
        enriched_output = await feature_generator.enrich_queries(queries, enrichment_formatter, sampled_data, conn)
        
        # Validate enriched output
        assert isinstance(enriched_output, pl.DataFrame)
        assert enriched_output.height > 0
        assert enriched_output.height == feature_generator.num_samples_for_enrichment
        
        # Check that all query names are in the enriched output
        for query in queries:
            assert query.name in enriched_output.columns, f'Query "{query.name}" not found in enriched output columns'
        
        # Check that enriched data contains values for each query
        for query in queries:
            query_data = enriched_output.select(pl.col(query.name)).drop_nulls()
            assert query_data.height > 0, f'Query "{query.name}" has no non-null values in enriched output'
            
            # Check that enriched data does not contain empty strings
            non_empty_data = query_data.filter(pl.col(query.name) != '')
            assert non_empty_data.height > 0, f'Query "{query.name}" has only empty string values in enriched output'

        logger.info(f'Enrichment successful: {enriched_output.height} rows with {len(enriched_output.columns)} columns')

        # 4. Test determine_dtype part
        updated_queries = await feature_generator.determine_dtypes(queries, enriched_output)
        
        # Validate updated queries
        assert isinstance(updated_queries, list)
        assert len(updated_queries) <= len(queries)  # Some queries might be filtered out
        assert all(isinstance(q, Query) for q in updated_queries)
        
        # Validate that all updated queries have valid dtypes
        valid_dtypes = [dtypes.boolean, dtypes.int32, dtypes.float64]
        for query in updated_queries:
            assert query.return_type is not None
            is_valid_simple_dtype = query.return_type in valid_dtypes
            is_valid_enum_dtype = isinstance(query.return_type, dtypes.EnumDtype)
            is_valid_dtype = is_valid_simple_dtype or is_valid_enum_dtype
            assert is_valid_dtype, f'Query "{query.name}" has invalid return_type: {query.return_type}'
            
            # Log the dtype determination results
            logger.info(f'Query "{query.name}" dtype determined as: {query.return_type}')
        
        logger.info(f'Dtype determination successful: {len(updated_queries)} queries with valid types')

        # 5. Test feature creation and evaluation
        features = [create_feature(
            query=query,
            formatter=enrichment_formatter,
            model=real_llm_with_spec)
            for query in updated_queries]
        
        # Validate features were created
        for feature in features:
            logger.info(f'Created feature: {feature.name} - {feature.description}')
        assert len(features) > 0 and len(main_dataset.data) > 0

        # 6. Test feature evaluation on first row
        for feature in features:
            strict_df = pl.DataFrame([main_dataset.data.get_column(col.name) for col in feature.params.cols])
            first_row_args = strict_df.row(0, named=False)
            # Evaluate the feature on the first row
            result = await feature.aevaluate(first_row_args, conn)

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
            
        logger.info('Feature creation and evaluation test completed successfully')
