"""Tests for ConversationActionRecommender (action recommender)."""

import json
import logging
import os
from pathlib import Path

import httpx
import polars as pl
import pytest
from cattrs import Converter
from duckdb import DuckDBPyConnection

from agentune.analyze.core.database import (
    DuckdbInMemoryDatabase,
    DuckdbManager,
)
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.llm import LLMContext, LLMSpec
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.core.types import EnumDtype
from agentune.analyze.feature.problem import ClassificationProblem, ProblemDescription
from agentune.analyze.feature.recommend import (
    ConversationActionRecommender,
    RecommendationsReport,
)
from agentune.analyze.feature.stats.base import FeatureWithFullStats
from agentune.analyze.run.base import RunContext

logger = logging.getLogger(__name__)


@pytest.fixture
async def real_llm_with_spec(httpx_async_client: httpx.AsyncClient) -> LLMWithSpec:
    """Create a real LLM for end-to-end testing."""
    llm_context = LLMContext(httpx_async_client)
    llm_spec = LLMSpec('openai', 'o3')
    llm_with_spec = LLMWithSpec(
        llm=llm_context.from_spec(llm_spec),
        spec=llm_spec
    )
    return llm_with_spec


@pytest.fixture
async def structuring_llm_with_spec(httpx_async_client: httpx.AsyncClient) -> LLMWithSpec:
    """Create a faster LLM for structuring (gpt-4o to avoid o3 timeouts and bugs)."""
    llm_context = LLMContext(httpx_async_client)
    llm_spec = LLMSpec('openai', 'gpt-4o')
    llm_with_spec = LLMWithSpec(
        llm=llm_context.from_spec(llm_spec),
        spec=llm_spec
    )
    return llm_with_spec


def _load_features_stats(file_path: Path) -> dict:
    """Load features and stats from JSON file."""
    with file_path.open() as f:
        return json.load(f)


def _extract_problem_info(features_and_stats: dict) -> tuple[str, str]:
    """Extract target column and desired outcome from features stats."""
    problem = features_and_stats['problem']
    target_column = problem['target_column']['name']
    desired_outcome = problem['problem_description'].get('target_desired_outcome', 'positive')
    return target_column, desired_outcome


def _load_main_dataset(csv_path: Path) -> Dataset:
    """Load main dataset from normalized CSV."""
    df = pl.read_csv(csv_path)
    
    # Create schema from the data
    unique_outcomes = tuple(sorted(df['outcome'].unique().drop_nulls().to_list()))
    schema = Schema(cols=(
        Field(name='id', dtype=EnumDtype(*[str(x) for x in df['id'].unique().to_list()])),
        Field(name='outcome', dtype=EnumDtype(*unique_outcomes)),
    ))
    
    return Dataset(schema=schema, data=df.select(['id', 'outcome']))


def _load_conversations_from_normalized_csv(csv_path: Path, conn: DuckDBPyConnection) -> None:
    """Load conversations from normalized CSV into DuckDB table."""
    df = pl.read_csv(csv_path)
    
    # Parse timestamps
    df = df.with_columns(
        pl.col('timestamp').str.to_datetime('%Y-%m-%dT%H:%M:%SZ')
    )
    
    # Register and create table
    conn.execute('DROP TABLE IF EXISTS conversations')
    conn.register('conversations_df', df)
    conn.execute('CREATE TABLE conversations AS SELECT * FROM conversations_df')
    conn.unregister('conversations_df')


def _reconstruct_features_with_stats(
    features_and_stats: dict,
    converter: Converter,
) -> list[FeatureWithFullStats]:
    """Reconstruct FeatureWithFullStats objects from JSON.
    
    Uses cattrs to deserialize the entire list. The conversation_strategy parameter
    is kept for potential future use if we need to patch database references.
    """
    # Use cattrs to deserialize the entire list
    return converter.structure(
        features_and_stats['features_with_train_stats'],
        list[FeatureWithFullStats]
    )


@pytest.mark.asyncio
async def test_action_recommender(
    real_llm_with_spec: LLMWithSpec,
    structuring_llm_with_spec: LLMWithSpec,
    run_context: RunContext,
) -> None:
    """End-to-end test of ConversationActionRecommender (action recommender) with real data."""
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        pytest.skip('OPENAI_API_KEY not set')
    
    # Load test data from centralized data directory (normalized CSVs)
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'conversations'
    main_csv_path = data_dir / 'example_main.csv'
    conversations_csv_path = data_dir / 'example_conversations_secondary.csv'
    features_file_path = data_dir / 'features_with_stats.json'
    
    # Get cattrs converter from run_context (as per docs/serialization.md)
    converter = run_context.ser_context.converter
    
    # Load features and extract problem info
    features_and_stats = _load_features_stats(features_file_path)
    target_column, desired_target_class = _extract_problem_info(features_and_stats)
    
    # Load main dataset from normalized CSV (much simpler!)
    dataset = _load_main_dataset(main_csv_path)
    
    ddb_manager = DuckdbManager(DuckdbInMemoryDatabase())
    with ddb_manager.cursor() as conn:
        # Load conversations from normalized CSV
        _load_conversations_from_normalized_csv(conversations_csv_path, conn)
        
        target_field = dataset.schema[target_column]
        classes_in_data = tuple(sorted(dataset.data[target_column].unique().drop_nulls().to_list()))
        
        problem_description = ProblemDescription(
            target_column=target_column,
            problem_type='classification',
            target_desired_outcome=desired_target_class,
        )
        
        problem = ClassificationProblem(
            problem_description=problem_description,
            target_column=target_field,
            classes=classes_in_data,
        )

    # Use cattrs converter for stats deserialization
    features_with_stats = _reconstruct_features_with_stats(
        features_and_stats,
        converter,
    )
    
    recommender = ConversationActionRecommender(
        model=real_llm_with_spec,
        structuring_model=structuring_llm_with_spec,
        num_samples=40,
        top_k_features=60,
    )
    
    with ddb_manager.cursor() as conn:
        report = await recommender.arecommend(
            problem=problem,
            features_with_stats=features_with_stats,
            dataset=dataset,
            conn=conn,
        )
    
    if report is None:
        pytest.skip('No conversation features found')
    
    assert isinstance(report, RecommendationsReport)
    assert len(report.analysis_summary) > 50
    assert len(report.recommendations) > 0
    
    # Optionally save results (useful for manual inspection)
    save_results = os.getenv('SAVE_TEST_RESULTS', 'false').lower() == 'true'
    if save_results:
        folder_for_results = Path.cwd() / 'tests_results'
        folder_for_results.mkdir(exist_ok=True)
        
        # Use cattrs to convert attrs to dict for JSON serialization
        report_dict = converter.unstructure(report)
        
        (folder_for_results / 'explainer_report.json').write_text(json.dumps(report_dict, indent=2))
        logger.info(f'Saved structured report to {folder_for_results / "explainer_report.json"}')
    
    logger.info('============= Final Results =============')
    logger.info(f'Dataset shape: {dataset.data.shape}')
    logger.info(f'Features analyzed: {len(features_with_stats)}')
    logger.info(f'Analysis summary: {report.analysis_summary[:200]}...')
    logger.info(f'Number of recommendations: {len(report.recommendations)}')
    
    # Verify structure
    for rec in report.recommendations:
        assert len(rec.title) > 0, 'Recommendation title should not be empty'
        assert len(rec.description) > 0, 'Recommendation description should not be empty'
        assert len(rec.rationale) > 0, 'Recommendation rationale should not be empty'
        
        # Verify that supporting features have SSE reduction values
        for feat_ref in rec.supporting_features:
            assert isinstance(feat_ref.sse_reduction, float), 'SSE reduction should be a float'
            # Note: SSE reduction can be negative (temporary errors), so we don't assert > 0
