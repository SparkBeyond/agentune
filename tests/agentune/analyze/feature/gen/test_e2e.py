import logging
import math
from pathlib import Path

import polars as pl
import pytest
from duckdb import DuckDBPyConnection

import agentune.core.types as dtypes
from agentune.analyze.feature.base import CategoricalFeature
from agentune.analyze.feature.gen.insightful_text_generator.features import create_feature
from agentune.analyze.feature.gen.insightful_text_generator.insightful_text_generator import (
    ConversationQueryFeatureGenerator,
)
from agentune.analyze.feature.gen.insightful_text_generator.schema import Query
from agentune.analyze.feature.problem import ClassificationProblem, ProblemDescription
from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.analyze.join.conversation import ConversationJoinStrategy
from agentune.core import duckdbio
from agentune.core.dataset import Dataset
from agentune.core.llm import LLMContext, LLMSpec
from agentune.core.schema import Field, Schema
from agentune.core.sercontext import LLMWithSpec

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


def _load_long_conversations_for_feature_gen(
    test_data_paths: dict[str, Path],
    conn: DuckDBPyConnection,
    duplication_factor: int = 10
) -> tuple[Dataset, str, TablesWithJoinStrategies, int, int]:
    """Load and prepare test data with very long conversations for token sampling testing.
    
    Args:
        test_data_paths: Paths to test data files
        conn: DuckDB connection
        duplication_factor: How many times to duplicate each conversation
        
    Returns:
        Tuple of (main_dataset, target_col, strategies, first_conv_id, last_conv_id)
    """
    # Load CSV files
    main_df = pl.read_csv(test_data_paths['main_csv'])
    conversations_df = pl.read_csv(test_data_paths['conversations_csv'])
    
    # Get the original conversation IDs for tracking
    original_ids = sorted(conversations_df['id'].unique().to_list())
    first_id = original_ids[0]
    last_id = original_ids[-1]
    
    # Parse timestamps first
    conversations_df = conversations_df.with_columns(
        pl.col('timestamp').str.to_datetime('%Y-%m-%dT%H:%M:%SZ')
    )
    
    # Duplicate the conversations multiple times to make them very long
    duplicated_dfs = []
    for i in range(duplication_factor):
        # Create a copy with offset timestamps to avoid duplicates
        df_copy = conversations_df.with_columns([
            pl.col('timestamp') + pl.duration(hours=i),
            # Add suffix to content to make it unique and longer
            pl.col('content') + f' [Duplicated message #{i + 1} - this makes the conversation much longer '
                                f'and should trigger token sampling when there are many conversations like this]'
        ])
        duplicated_dfs.append(df_copy)
    
    # Concatenate all duplicated conversations
    long_conversations_df = pl.concat(duplicated_dfs, how='vertical')
    
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
    # convert columns to appropriate types (timestamps already converted above)
    for field in secondary_schema.cols:
        if field.name != 'timestamp':  # Already converted
            long_conversations_df = long_conversations_df.with_columns(pl.col(field.name).cast(field.dtype.polars_type))

    # Create datasets
    main_dataset = Dataset(schema=main_schema, data=main_df)
    secondary_dataset = Dataset(schema=secondary_schema, data=long_conversations_df)

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

    return main_dataset, 'outcome', strategies, first_id, last_id


@pytest.fixture
async def real_llm_with_spec(llm_context_nocache: LLMContext) -> LLMWithSpec:
    """Create a real LLM for end-to-end testing."""
    llm_spec = LLMSpec('openai', 'gpt-4o-mini')  # Use a smaller, faster model for testing
    llm_with_spec = LLMWithSpec(
        llm=llm_context_nocache.from_spec(llm_spec),
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
        query_enrich_model=real_llm_with_spec,
        max_samples_for_generation=10,
        num_samples_for_enrichment=5,
        num_features_per_round=5,
        num_actionable_rounds=2,
        num_creative_features=5,
        random_seed=random_seed
    )

    # imitate the analysis
    conversation_strategies = feature_generator.find_conversation_strategies(strategies)

    for conversation_strategy in conversation_strategies:

        # 1. Generate queries using two-phase approach
        queries = await feature_generator._generate_queries(conversation_strategy, main_dataset, problem, conn)

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

        # 2. Test enrichment part
        sampler = feature_generator._get_sampler(problem)
        sampled_data = sampler.sample(main_dataset, feature_generator.num_samples_for_enrichment, feature_generator.random_seed)
        enrichment_formatter = feature_generator._get_formatter(conversation_strategy, problem, generation_mode=False)
        enriched_results = await feature_generator.enrich_queries(queries, enrichment_formatter, sampled_data, conn)
        
        # Validate enriched output
        assert isinstance(enriched_results, list)
        assert len(enriched_results) > 0
        assert len(enriched_results) <= len(queries)  # May be fewer if some failed
        
        # Check that all results have the correct structure
        for result in enriched_results:
            assert result.query in queries
            assert len(result.enriched_values) == feature_generator.num_samples_for_enrichment
            # Check that enriched data contains some non-null, non-empty values
            non_null_values = [v for v in result.enriched_values if v is not None and v != '']
            assert len(non_null_values) > 0, f'Query "{result.query.name}" has only null/empty values in enriched output'

        logger.info(f'Enrichment successful: {len(enriched_results)} queries enriched')

        # 3. Test determine_dtype part
        updated_queries = await feature_generator.determine_dtypes(enriched_results)
        
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

        # 4. Test feature creation and computation
        features = [create_feature(
            query=query,
            formatter=enrichment_formatter,
            model=real_llm_with_spec)
            for query in updated_queries]
        
        # Validate features were created
        for feature in features:
            logger.info(f'Created feature: {feature.name} - {feature.description}')
        assert len(features) > 0 and len(main_dataset.data) > 0

        # 5. Test feature compute on first row
        for feature in features:
            strict_df = pl.DataFrame([main_dataset.data.get_column(col.name) for col in feature.params.cols])
            first_row_args = strict_df.row(0, named=False)
            # Compute the feature on the first row
            compute_result = await feature.acompute(first_row_args, conn)

            # Validate result
            if compute_result is None:
                logger.info(f'Feature {feature.name} returned None (missing value)')
            else:
                # Check that the type of the result matches the expected Python type from DuckDB
                expected_python_type = dtypes.python_type_from_polars(feature.dtype)
                assert isinstance(compute_result, expected_python_type), f'Feature {feature.name} returned {type(compute_result)} but expected {expected_python_type}'
                
                # Additional type-specific validation
                if feature.dtype.is_numeric():
                    # For numeric types, check that the result is not NaN or infinite
                    assert isinstance(compute_result, float | int), f'Feature {feature.name} returned non-numeric type {type(compute_result)}'
                    assert not math.isnan(float(compute_result)), f'Feature {feature.name} returned NaN'
                    assert not math.isinf(float(compute_result)), f'Feature {feature.name} returned infinite value'
                elif isinstance(feature, CategoricalFeature):
                    # For categorical/enum types, check that the result is one of the valid categories
                    valid_categories = [*list(feature.categories), CategoricalFeature.other_category]
                    assert compute_result in valid_categories, (
                        f'Feature {feature.name} returned "{compute_result}" which is not in valid categories: '
                        f'{feature.categories}'
                    )
                
                logger.info(f'Feature {feature.name} compute successful: {compute_result} (type: {type(compute_result).__name__})')
            
        logger.info('Feature creation and compute test completed successfully')


@pytest.mark.integration
async def test_feature_generation_with_long_conversations_token_sampling(
    test_data_conversations: dict[str, Path],
    conn: DuckDBPyConnection,
    real_llm_with_spec: LLMWithSpec,
    problem: ClassificationProblem
) -> None:
    """Test feature generation with very long conversations to verify token sampling works.
    
    This test creates artificially long conversations by duplicating the original data,
    then verifies that:
    1. No token limit errors occur during query generation
    2. Feature generation completes successfully despite long conversations
    3. The system gracefully handles token limits through sampling
    """
    # Load test data with very long conversations (duplicated 5 times to make them long but not too extreme)
    main_dataset, target_col, strategies, first_conv_id, last_conv_id = _load_long_conversations_for_feature_gen(
        test_data_conversations, conn, duplication_factor=5
    )
    
    logger.info(f'Created very long conversations - first ID: {first_conv_id}, last ID: {last_conv_id}')
    
    # Create feature generator with more samples initially to force token sampling
    feature_generator = ConversationQueryFeatureGenerator(
        query_generator_model=real_llm_with_spec,
        max_samples_for_generation=25,  # Start with 25 to force token reduction down to ~10-15
        num_samples_for_enrichment=5,
        num_features_per_round=3,  # Smaller to focus on the token sampling behavior
        num_actionable_rounds=1,  # Reduce to make test faster
        num_creative_features=2,
        query_enrich_model=real_llm_with_spec,
        random_seed=42
    )
    
    # Test the query generation step specifically (where token sampling occurs)
    conversation_strategies = feature_generator.find_conversation_strategies(strategies)
    assert len(conversation_strategies) > 0, 'Should find conversation strategies'
    
    conversation_strategy = conversation_strategies[0]
    
    # This should NOT raise any token limit errors due to the token sampling logic
    logger.info('Testing query generation with very long conversations...')
    queries = await feature_generator._generate_queries(conversation_strategy, main_dataset, problem, conn)
    
    # Validate that queries were generated successfully
    assert isinstance(queries, list)
    assert len(queries) > 0, 'Should generate at least one query despite long conversations'
    assert all(isinstance(q, Query) for q in queries)
    
    logger.info(f'Successfully generated {len(queries)} queries with token sampling')
    
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
        
        logger.info(f'Generated query: {query.name} - {query.query_text[:100]}...')
    
    # Test enrichment step to make sure the full pipeline works
    logger.info('Testing enrichment step...')
    sampler = feature_generator._get_sampler(problem)
    sampled_data = sampler.sample(main_dataset, feature_generator.num_samples_for_enrichment, feature_generator.random_seed)
    enrichment_formatter = feature_generator._get_formatter(conversation_strategy, problem, generation_mode=False)
    
    # This should also work without token errors
    enriched_results = await feature_generator.enrich_queries(queries, enrichment_formatter, sampled_data, conn)
    
    # Validate enriched output
    assert isinstance(enriched_results, list)
    assert len(enriched_results) > 0
    assert len(enriched_results) <= len(queries)
    for result in enriched_results:
        assert len(result.enriched_values) == feature_generator.num_samples_for_enrichment
    
    logger.info(f'Enrichment successful: {len(enriched_results)} queries enriched')
    
    # Test dtype determination
    logger.info('Testing dtype determination...')
    updated_queries = await feature_generator.determine_dtypes(enriched_results)
    
    # Validate updated queries
    assert isinstance(updated_queries, list)
    assert len(updated_queries) <= len(queries)  # Some queries might be filtered out
    
    logger.info(f'Dtype determination successful: {len(updated_queries)} queries with valid types')
    
    # Test feature creation
    logger.info('Testing feature creation...')
    features = [create_feature(
        query=query,
        formatter=enrichment_formatter,
        model=real_llm_with_spec)
        for query in updated_queries]
    
    assert len(features) > 0, 'Should create at least one feature'
    
    # Test feature computation on first row
    logger.info('Testing feature computation...')
    for feature in features:
        strict_df = pl.DataFrame([main_dataset.data.get_column(col.name) for col in feature.params.cols])
        first_row_args = strict_df.row(0, named=False)
        
        # This should work without errors
        compute_result = await feature.acompute(first_row_args, conn)
        
        logger.info(f'Feature {feature.name} computed successfully: {compute_result}')
    
    logger.info('============= Token Sampling Test Results =============')
    logger.info(f'Dataset shape: {main_dataset.data.shape}')
    logger.info(f'Generated queries: {len(queries)}')
    logger.info(f'Final features: {len(features)}')
    logger.info('Test completed successfully - no token limit errors occurred!')
    logger.info('Feature generation pipeline handled very long conversations gracefully through token sampling!')


@pytest.mark.integration
async def test_feature_generation_with_extremely_long_conversations_error_case(
    test_data_conversations: dict[str, Path],
    conn: DuckDBPyConnection,
    real_llm_with_spec: LLMWithSpec,
    problem: ClassificationProblem
) -> None:
    """Test that feature generation properly handles the error case when conversations are too long.
    
    This test creates extremely long conversations that can't fit even with minimum samples,
    and verifies that the system raises a clear error instead of crashing.
    """
    # Load test data with extremely long conversations (duplicated 25 times to make them way too long)
    main_dataset, target_col, strategies, first_conv_id, last_conv_id = _load_long_conversations_for_feature_gen(
        test_data_conversations, conn, duplication_factor=25  # Very large duplication factor
    )
    
    logger.info(f'Created extremely long conversations - first ID: {first_conv_id}, last ID: {last_conv_id}')
    
    # Create feature generator with even smaller samples
    feature_generator = ConversationQueryFeatureGenerator(
        query_generator_model=real_llm_with_spec,
        max_samples_for_generation=10, 
        num_samples_for_enrichment=3,
        num_features_per_round=2,
        num_actionable_rounds=1,
        num_creative_features=1,
        query_enrich_model=real_llm_with_spec,
        random_seed=42
    )
    
    # Test the query generation step - this should raise a clear error
    conversation_strategies = feature_generator.find_conversation_strategies(strategies)
    assert len(conversation_strategies) > 0, 'Should find conversation strategies'
    
    conversation_strategy = conversation_strategies[0]
    
    logger.info('Testing query generation with extremely long conversations (expecting error)...')
    
    # This should raise a ValueError with a clear message about token limits
    with pytest.raises(ValueError, match='Cannot fit the minimal amount of needed sample conversations.*within the LLM token limit.*Try providing shorter conversations as input'):
        await feature_generator._generate_queries(conversation_strategy, main_dataset, problem, conn)
    
    logger.info('✅ Correctly raised ValueError for extremely long conversations that exceed token limits!')


@pytest.mark.integration
async def test_enrich_queries_returns_enriched_query_results(
    test_dataset_with_strategy: tuple[Dataset, str, TablesWithJoinStrategies],
    conn: DuckDBPyConnection,
    real_llm_with_spec: LLMWithSpec,
    problem: ClassificationProblem
) -> None:
    """Test that enrich_queries returns EnrichedQueryResult objects with proper structure.
    
    This integration test validates that:
    1. enrich_queries returns list[EnrichedQueryResult] not DataFrame
    2. Each result pairs a query with its enriched values
    3. The pipeline works end-to-end with the new structure
    4. Results can be fewer than input queries (graceful degradation)
    """
    main_dataset, target_col, strategies = test_dataset_with_strategy
    
    feature_generator = ConversationQueryFeatureGenerator(
        query_generator_model=real_llm_with_spec,
        query_enrich_model=real_llm_with_spec,
        max_samples_for_generation=10,
        num_samples_for_enrichment=3,  # Small number for faster test
        num_features_per_round=2,
        num_actionable_rounds=1,
        num_creative_features=0,
        random_seed=42
    )
    
    conversation_strategies = feature_generator.find_conversation_strategies(strategies)
    conversation_strategy = conversation_strategies[0]
    
    # Generate some queries
    queries = await feature_generator._generate_queries(conversation_strategy, main_dataset, problem, conn)
    assert len(queries) >= 1, 'Need at least 1 query for this test'
    
    # Take first 2 queries for the test
    test_queries = queries[:min(2, len(queries))]
    
    sampler = feature_generator._get_sampler(problem)
    sampled_data = sampler.sample(main_dataset, feature_generator.num_samples_for_enrichment, feature_generator.random_seed)
    enrichment_formatter = feature_generator._get_formatter(conversation_strategy, problem, generation_mode=False)
    
    # Test the new enrich_queries behavior
    enriched_results = await feature_generator.enrich_queries(test_queries, enrichment_formatter, sampled_data, conn)
    
    # Validate results structure
    logger.info(f'Enrichment completed: {len(enriched_results)} results from {len(test_queries)} queries')
    
    # Should return a list of EnrichedQueryResult
    assert isinstance(enriched_results, list), 'Should return a list'
    assert len(enriched_results) > 0, 'Should have at least one successful enrichment'
    assert len(enriched_results) <= len(test_queries), 'Results should not exceed input queries'
    
    # Check that each result has proper structure
    for result in enriched_results:
        # Each result should be an EnrichedQueryResult
        assert hasattr(result, 'query'), 'Result should have query attribute'
        assert hasattr(result, 'enriched_values'), 'Result should have enriched_values attribute'
        
        # Query should be one of the input queries
        assert result.query in test_queries, f'Result query {result.query.name} should be in input queries'
        
        # Enriched values should have correct length
        assert len(result.enriched_values) == feature_generator.num_samples_for_enrichment, \
            f'Enriched values should have {feature_generator.num_samples_for_enrichment} items'
        
        # Successful results should have some non-None values
        non_none_values = [v for v in result.enriched_values if v is not None]
        assert len(non_none_values) > 0, f'Result for {result.query.name} should have some non-None values'
    
    # The pipeline should continue to work with the enriched results
    updated_queries = await feature_generator.determine_dtypes(enriched_results)
    assert len(updated_queries) >= 1, 'Should have at least one query with determined dtype'
    assert all(q.return_type is not None for q in updated_queries), 'All updated queries should have return types'
    
    logger.info(f'✅ Pipeline works with new EnrichedQueryResult structure: {len(updated_queries)} queries with valid dtypes')
