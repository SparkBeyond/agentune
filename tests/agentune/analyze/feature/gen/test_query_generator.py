import logging
from pathlib import Path

import httpx
import polars as pl
import pytest
from duckdb import DuckDBPyConnection

import agentune.analyze.core.types as dtypes
from agentune.analyze.context.base import TablesWithContextDefinitions
from agentune.analyze.context.conversation import ConversationContext
from agentune.analyze.core import duckdbio
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.llm import LLMContext, LLMSpec
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.feature.gen.insightful_text_generator.insightful_text_generator import (
    ConversationQueryFeatureGenerator,
)
from agentune.analyze.feature.gen.insightful_text_generator.schema import Query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def test_data_paths() -> dict[str, Path]:
    """Get the test data directory."""
    test_data_dir = Path(__file__).parent.parent.parent / 'data' / 'conversations'
    paths_dict = {
        'main_csv': test_data_dir / 'example_main.csv',
        'conversations_csv': test_data_dir / 'example_conversations_context.csv'
    }
    return paths_dict


@pytest.fixture
def test_dataset_with_context(test_data_paths: dict[str, Path], conn: DuckDBPyConnection) -> tuple[Dataset, str, TablesWithContextDefinitions]:
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

    # Create conversation context schema
    context_schema = Schema((
        Field(name='id', dtype=dtypes.int32),
        Field(name='timestamp', dtype=dtypes.timestamp),
        Field(name='role', dtype=dtypes.string),
        Field(name='content', dtype=dtypes.string),
    ))
    # convert columns to appropriate types
    conversations_df = conversations_df.with_columns(
        pl.col('timestamp').str.to_datetime('%Y-%m-%dT%H:%M:%SZ')
    )
    for field in context_schema.cols:
        conversations_df = conversations_df.with_columns(pl.col(field.name).cast(field.dtype.polars_type))

    # Create datasets
    main_dataset = Dataset(schema=main_schema, data=main_df)
    context_dataset = Dataset(schema=context_schema, data=conversations_df)

    # Ingest tables
    duckdbio.ingest(conn, 'main', main_dataset.as_source())
    context_table = duckdbio.ingest(conn, 'conversations', context_dataset.as_source())

    # Create conversation context
    conversation_context = ConversationContext[int].on_table(
        'conversations',
        context_table.table,
        'id',           # main_table_id_column
        'id',           # id_column
        'timestamp',    # timestamp_column
        'role',         # role_column
        'content'       # content_column
    )

    # Create index
    conversation_context.index.create(conn, context_table.table.name, if_not_exists=True)

    # Create context definitions
    contexts = TablesWithContextDefinitions.group([conversation_context])

    return main_dataset, 'outcome', contexts

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



@pytest.mark.integration
async def test_end_to_end_pipeline_with_real_llm(test_dataset_with_context: tuple[Dataset, str, TablesWithContextDefinitions], conn: DuckDBPyConnection,
                                                 real_llm_with_spec: LLMWithSpec) -> None:
    """Test the complete end-to-end pipeline with real LLM."""
    main_dataset, target_col, contexts = test_dataset_with_context
    random_seed = 42  # Use a fixed seed for reproducibility

    feature_generator: ConversationQueryFeatureGenerator = ConversationQueryFeatureGenerator(
        query_generator_model=real_llm_with_spec,
        num_features_to_generate=5,
        num_samples_for_generation=10,
        num_samples_for_enrichment=5,
        field_descriptions='id: unique identifier, outcome: resolution status',
        what_is_an_instance='a customer support conversation',
        instance_description='Each instance represents a complete customer support conversation with multiple messages between customer and agent',
        target_value='resolved',
        query_enrich_model=real_llm_with_spec,
        random_seed=random_seed
    )

    # imitate the feature search
    target_field = main_dataset.schema[target_col]
    conversation_contexts = feature_generator.find_conversation_contexts(contexts)

    for conversation_context in conversation_contexts:
        # 1. Create a query generator for the conversation context
        query_generator = feature_generator.create_query_generator(conversation_context, target_field)

        # 2. Generate queries from the conversation data
        queries = await query_generator.agenerate_queries(main_dataset, conn, random_seed=feature_generator.random_seed)

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
