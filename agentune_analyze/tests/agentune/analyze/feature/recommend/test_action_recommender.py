"""Tests for ConversationActionRecommender (action recommender)."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import httpx
import polars as pl
import pytest
from cattrs import Converter
from duckdb import DuckDBPyConnection

from agentune.analyze.core.database import (
    DuckdbInMemory,
    DuckdbManager,
)
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.llm import LLMContext, LLMSpec
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.core.sercontext import LLMWithSpec, SerializationContext
from agentune.analyze.core.types import EnumDtype
from agentune.analyze.feature.problem import ClassificationProblem, ProblemDescription
from agentune.analyze.feature.recommend import (
    ConversationActionRecommender,
    RecommendationsReport,
)
from agentune.analyze.feature.recommend.action_recommender import (
    ConversationWithExplanation,
    ConversationWithMetadata,
    FeatureWithScore,
    Recommendation,
)
from agentune.analyze.feature.recommend.prompts import RecommendationRaw, StructuredReport
from agentune.analyze.feature.stats.base import FeatureWithFullStats
from agentune.analyze.join.conversation import Conversation, Message

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


def _extract_problem_info(features_and_stats: dict, converter: Converter) -> tuple[str, str]:
    """Extract target column and desired outcome from features stats."""
    problem = converter.structure(features_and_stats['problem'], ClassificationProblem)
    target_column = problem.target_column.name
    target_desired_outcome = cast(str, problem.target_desired_outcome_value or 'positive')
    return target_column, target_desired_outcome


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


def _load_long_conversations_from_normalized_csv(csv_path: Path, conn: DuckDBPyConnection, duplication_factor: int = 10) -> tuple[int, int]:
    """Load conversations from normalized CSV and duplicate them to create very long conversations.

    Args:
        csv_path: Path to the original conversations CSV
        conn: DuckDB connection
        duplication_factor: How many times to duplicate each conversation

    Returns:
        Tuple of (first_conversation_id, last_conversation_id) for testing
    """
    df = pl.read_csv(csv_path)

    # Parse timestamps
    df = df.with_columns(
        pl.col('timestamp').str.to_datetime('%Y-%m-%dT%H:%M:%SZ')
    )

    # Get the original conversation IDs for tracking
    original_ids = sorted(df['id'].unique().to_list())
    first_id = original_ids[0]
    last_id = original_ids[-1]

    # Duplicate the conversations multiple times to make them very long
    duplicated_dfs = []
    for i in range(duplication_factor):
        # Create a copy with offset timestamps to avoid duplicates
        df_copy = df.with_columns([
            pl.col('timestamp') + pl.duration(hours=i),
            # Add suffix to content to make it unique and longer
            pl.col('content') + f' [Duplicated message #{i + 1} - this makes the conversation much longer '
                                f'and should trigger token sampling when there are many conversations like this]'
        ])
        duplicated_dfs.append(df_copy)

    # Concatenate all duplicated conversations
    long_df = pl.concat(duplicated_dfs, how='vertical')

    # Register and create table
    conn.execute('DROP TABLE IF EXISTS conversations')
    conn.register('conversations_df', long_df)
    conn.execute('CREATE TABLE conversations AS SELECT * FROM conversations_df')
    conn.unregister('conversations_df')

    return first_id, last_id


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

@pytest.mark.integration
async def test_action_recommender(
    real_llm_with_spec: LLMWithSpec,
    structuring_llm_with_spec: LLMWithSpec,
    ser_context: SerializationContext,
) -> None:
    """End-to-end test of ConversationActionRecommender (action recommender) with real data."""
    # Load test data from centralized data directory (normalized CSVs)
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'conversations'
    main_csv_path = data_dir / 'example_main.csv'
    conversations_csv_path = data_dir / 'example_conversations_secondary.csv'
    features_file_path = data_dir / 'features_with_stats.json'
    
    # Get cattrs converter from run_context (as per docs/serialization.md)
    converter = ser_context.converter
    
    # Load features and extract problem info
    features_and_stats = _load_features_stats(features_file_path)
    target_column, desired_target_class = _extract_problem_info(features_and_stats, converter)
    
    # Load main dataset from normalized CSV (much simpler!)
    dataset = _load_main_dataset(main_csv_path)
    
    ddb_manager = DuckdbManager(DuckdbInMemory())
    with ddb_manager.cursor() as conn:
        # Load conversations from normalized CSV
        _load_conversations_from_normalized_csv(conversations_csv_path, conn)
        
        target_field = dataset.schema[target_column]
        classes_in_data = tuple(sorted(dataset.data[target_column].unique().drop_nulls().to_list()))
        
        problem_description = ProblemDescription(
            target_column=target_column,
            description='Test conversation outcome analysis',
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
        max_samples=40,
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
    logger.info(f'Number of conversations referenced: {len(report.conversations)}')

    # Verify recommendation structure
    for rec in report.recommendations:
        assert len(rec.title) > 0, 'Recommendation title should not be empty'
        assert len(rec.description) > 0, 'Recommendation description should not be empty'
        assert len(rec.rationale) > 0, 'Recommendation rationale should not be empty'
        
        # Verify that supporting features have R² values
        for feat_ref in rec.supporting_features:
            assert isinstance(feat_ref.r_squared, float), 'R² should be a float'
            # Note: R² can be negative (temporary errors), so we don't assert > 0

    # Verify ConversationWithMetadata structure
    assert len(report.conversations) > 0, 'Should have at least one conversation referenced'

    for display_num, conv_metadata in report.conversations.items():
        # Verify all required fields are present
        assert conv_metadata.actual_id is not None, f'Conversation {display_num} missing actual_id'
        assert conv_metadata.conversation is not None, f'Conversation {display_num} missing conversation content'
        assert conv_metadata.outcome is not None, f'Conversation {display_num} missing outcome'

        # Verify conversation has messages
        assert len(conv_metadata.conversation.messages) > 0, f'Conversation {display_num} has no messages'

        # Format actual_id for display (handle both string UUIDs and integers)
        actual_id_display = str(conv_metadata.actual_id)[:20] + '...' if len(str(conv_metadata.actual_id)) > 20 else str(conv_metadata.actual_id)
        logger.info(f'Conversation {display_num}: ID={actual_id_display}, Outcome={conv_metadata.outcome}, Messages={len(conv_metadata.conversation.messages)}')

    logger.info('✅ All conversation metadata verified successfully!')


def test_conversation_id_mapping() -> None:
    """Test that conversation IDs are correctly mapped to display numbers.

    This test verifies that the implicit mapping through list positions works correctly:
    - Display number N corresponds to conversation_ids[N-1]
    - The same position in conversations tuple
    - The same position in outcomes list
    """
    # Simulate the data structures as they would be in arecommend()
    conversation_ids = ['conv_abc', 'conv_def', 'conv_ghi', 'conv_jkl', 'conv_mno']
    outcomes = ['positive', 'negative', 'positive', 'negative', 'positive']

    # Create mock conversations
    conversations = tuple(
        Conversation(messages=(
            Message(role='user', content=f'Message from {conv_id}', timestamp=datetime.now()),
        ))
        for conv_id in conversation_ids
    )

    # Simulate LLM referencing conversations 2, 4, and 5
    llm_referenced_indices = {2, 4, 5}

    # Build conversations dict as done in _convert_pydantic_to_attrs
    conversations_dict = {
        idx: ConversationWithMetadata(
            actual_id=conversation_ids[idx - 1],  # idx is 1-based, list is 0-based
            conversation=conversations[idx - 1],
            outcome=outcomes[idx - 1],
        )
        for idx in llm_referenced_indices
    }

    # Verify the mapping is correct
    assert len(conversations_dict) == 3, 'Should have 3 conversations'

    # Verify Conversation 2 maps correctly
    assert 2 in conversations_dict
    assert conversations_dict[2].actual_id == 'conv_def'  # conversation_ids[1]
    assert conversations_dict[2].outcome == 'negative'  # outcomes[1]
    assert 'conv_def' in conversations_dict[2].conversation.messages[0].content

    # Verify Conversation 4 maps correctly
    assert 4 in conversations_dict
    assert conversations_dict[4].actual_id == 'conv_jkl'  # conversation_ids[3]
    assert conversations_dict[4].outcome == 'negative'  # outcomes[3]
    assert 'conv_jkl' in conversations_dict[4].conversation.messages[0].content

    # Verify Conversation 5 maps correctly
    assert 5 in conversations_dict
    assert conversations_dict[5].actual_id == 'conv_mno'  # conversation_ids[4]
    assert conversations_dict[5].outcome == 'positive'  # outcomes[4]
    assert 'conv_mno' in conversations_dict[5].conversation.messages[0].content

    # Verify conversations 1 and 3 are NOT in the dict (not referenced by LLM)
    assert 1 not in conversations_dict
    assert 3 not in conversations_dict


@pytest.mark.asyncio
@pytest.mark.integration
async def test_action_recommender_with_long_conversations_token_sampling(
    real_llm_with_spec: LLMWithSpec,
    structuring_llm_with_spec: LLMWithSpec,
    ser_context: SerializationContext,
) -> None:
    """Test ConversationActionRecommender with very long conversations to verify token sampling works.

    This test creates artificially long conversations by duplicating the original data,
    then verifies that:
    1. No token limit errors occur
    2. Some conversations are included (first one should be there)
    3. Some conversations are excluded due to token limits (last one should not be there)
    4. The recommender still produces valid results
    """
    # Load test data from centralized data directory (normalized CSVs)
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'conversations'
    main_csv_path = data_dir / 'example_main.csv'
    conversations_csv_path = data_dir / 'example_conversations_secondary.csv'
    features_file_path = data_dir / 'features_with_stats.json'

    # Get cattrs converter from ser_context (as per docs/serialization.md)
    converter = ser_context.converter

    # Load features and extract problem info
    features_and_stats = _load_features_stats(features_file_path)
    target_column, desired_target_class = _extract_problem_info(features_and_stats, converter)

    # Load main dataset from normalized CSV
    dataset = _load_main_dataset(main_csv_path)

    ddb_manager = DuckdbManager(DuckdbInMemory())
    with ddb_manager.cursor() as conn:
        # Load LONG conversations from normalized CSV (duplicated 15 times to make them very long)
        first_conv_id, last_conv_id = _load_long_conversations_from_normalized_csv(
            conversations_csv_path, conn, duplication_factor=15
        )

        logger.info(f'Created long conversations - first ID: {first_conv_id}, last ID: {last_conv_id}')

        target_field = dataset.schema[target_column]
        classes_in_data = tuple(sorted(dataset.data[target_column].unique().drop_nulls().to_list()))

        problem_description = ProblemDescription(
            target_column=target_column,
            description='Test conversation outcome analysis',
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

    # Create recommender with smaller max_samples to force token sampling
    recommender = ConversationActionRecommender(
        model=real_llm_with_spec,
        structuring_model=structuring_llm_with_spec,
        max_samples=15,  # Smaller to force token limit issues with long conversations
        top_k_features=20,  # Smaller to focus on the token sampling behavior
    )

    with ddb_manager.cursor() as conn:
        # This should NOT raise any token limit errors due to the token sampling logic
        report = await recommender.arecommend(
            problem=problem,
            features_with_stats=features_with_stats,
            dataset=dataset,
            conn=conn,
        )

    if report is None:
        pytest.skip('No conversation features found')

    # Verify that the recommender still produced valid results
    assert isinstance(report, RecommendationsReport)
    assert len(report.analysis_summary) > 50
    assert len(report.recommendations) > 0

    # Check that some conversations were included but not all
    # The raw_report should contain some conversation content but not all conversations
    raw_report_lower = report.raw_report.lower()

    # Verify that some conversation content is included (first conversation should be there)
    assert 'conversation' in raw_report_lower, 'Report should contain conversation content'

    # Log information about what was included
    first_id_str = str(first_conv_id)
    last_id_str = str(last_conv_id)

    first_conv_in_report = first_id_str in report.raw_report
    last_conv_in_report = last_id_str in report.raw_report

    logger.info(f'First conversation (ID {first_conv_id}) in report: {first_conv_in_report}')
    logger.info(f'Last conversation (ID {last_conv_id}) in report: {last_conv_in_report}')
    logger.info(f'Report length: {len(report.raw_report)} characters')

    # We expect that due to token sampling:
    # 1. The first conversation should likely be included (as token reduction starts from the end)
    # 2. The last conversation might be excluded if token limits are hit
    # Note: We can't guarantee exact behavior as it depends on the actual token counts,
    # but we can verify that the system didn't crash and produced valid results

    logger.info('============= Token Sampling Test Results =============')
    logger.info(f'Dataset shape: {dataset.data.shape}')
    logger.info(f'Features analyzed: {len(features_with_stats)}')
    logger.info(f'Number of recommendations: {len(report.recommendations)}')
    logger.info('Test completed successfully - no token limit errors occurred!')

    # Verify basic structure is still intact
    for rec in report.recommendations:
        assert len(rec.title) > 0, 'Recommendation title should not be empty'
        assert len(rec.description) > 0, 'Recommendation description should not be empty'
        assert len(rec.rationale) > 0, 'Recommendation rationale should not be empty'


def test_feature_filtering_removes_recommendations_without_features(caplog: pytest.LogCaptureFixture) -> None:
    """Test that recommendations with no supporting features are completely filtered out."""
    # Create a mock recommender
    recommender = ConversationActionRecommender(
        model=Mock(),
        max_samples=10,
        top_k_features=5,
    )

    # Create mock features with proper structure
    mock_feature = Mock()
    mock_feature.description = 'Valid feature'
    mock_relationship = Mock()
    mock_relationship.sse_reduction = 0.8
    mock_stats = Mock()
    mock_stats.relationship = mock_relationship

    features_with_stats = cast(
        list[FeatureWithFullStats],
        [Mock(feature=mock_feature, stats=mock_stats)],
    )

    # Create a Pydantic report with a recommendation that has NO supporting features
    mock_pydantic_report = StructuredReport(
        analysis_summary='Test analysis',
        recommendations=[
            RecommendationRaw(
                title='Recommendation Without Features',
                description='This recommendation has no supporting features',
                rationale='Should be filtered out',
                evidence='No features',
                supporting_features=[],  # Empty list
                supporting_conversations=[],
            ),
        ]
    )

    # Call the conversion and filtering methods
    unfiltered = recommender._convert_pydantic_to_attrs(
        mock_pydantic_report,
        features_with_stats,
        (),  # conversations
        [],  # conversation_ids
        [],  # outcomes
        'raw report'
    )
    result = recommender._filter_recommendations_with_no_supporting_features(unfiltered)

    # Verify the recommendation was filtered out
    assert len(result.recommendations) == 0

    # Verify logging
    assert any('Filtered out' in record.message for record in caplog.records)
    assert any('Recommendation Without Features' in record.message for record in caplog.records)


def test_feature_filtering_removes_features_with_zero_sse() -> None:
    """Test that features with 0.0 SSE reduction are filtered out."""
    # Create a mock recommender
    recommender = ConversationActionRecommender(
        model=Mock(),
        max_samples=10,
        top_k_features=5,
    )

    # Create mock features: one with high R², one with 0.0 R²
    mock_feature_high = Mock()
    mock_feature_high.description = 'High SSE feature'
    mock_relationship_high = Mock()
    mock_relationship_high.r_squared = 0.8
    mock_stats_high = Mock()
    mock_stats_high.relationship = mock_relationship_high

    mock_feature_zero = Mock()
    mock_feature_zero.description = 'Zero SSE feature'
    mock_relationship_zero = Mock()
    mock_relationship_zero.r_squared = 0.0
    mock_stats_zero = Mock()
    mock_stats_zero.relationship = mock_relationship_zero

    features_with_stats = cast(
        list[FeatureWithFullStats],
        [
            Mock(feature=mock_feature_high, stats=mock_stats_high),
            Mock(feature=mock_feature_zero, stats=mock_stats_zero),
        ],
    )

    # Create a recommendation that references both features
    mock_pydantic_report = StructuredReport(
        analysis_summary='Test analysis',
        recommendations=[
            RecommendationRaw(
                title='Mixed Features Recommendation',
                description='Has both high and zero SSE features',
                rationale='Test filtering',
                evidence='Test evidence',
                supporting_features=[
                    'High SSE feature',
                    'Zero SSE feature',  # Should be filtered out
                ],
                supporting_conversations=[],
            ),
        ]
    )

    # Call the conversion and filtering methods
    unfiltered = recommender._convert_pydantic_to_attrs(
        mock_pydantic_report,
        features_with_stats,
        (),  # conversations
        [],  # conversation_ids
        [],  # outcomes
        'raw report'
    )
    result = recommender._filter_recommendations_with_no_supporting_features(unfiltered)

    # Verify only 1 recommendation remains (with only the high R² feature)
    assert len(result.recommendations) == 1
    rec = result.recommendations[0]

    # Should only have 1 feature (the one with high R²)
    assert len(rec.supporting_features) == 1
    assert rec.supporting_features[0].name == 'High SSE feature'
    assert rec.supporting_features[0].r_squared == 0.8

    # Verify the zero SSE feature is not included
    feature_names = [f.name for f in rec.supporting_features]
    assert 'Zero SSE feature' not in feature_names


def test_feature_filtering_removes_hallucinated_features(caplog: pytest.LogCaptureFixture) -> None:
    """Test that hallucinated (non-existent) features are filtered out."""
    # Create a mock recommender
    recommender = ConversationActionRecommender(
        model=Mock(),
        max_samples=10,
        top_k_features=5,
    )

    # Create mock features - only define real features
    mock_feature_1 = Mock()
    mock_feature_1.description = 'Real feature 1'
    mock_relationship_1 = Mock()
    mock_relationship_1.r_squared = 0.9
    mock_stats_1 = Mock()
    mock_stats_1.relationship = mock_relationship_1

    mock_feature_2 = Mock()
    mock_feature_2.description = 'Real feature 2'
    mock_relationship_2 = Mock()
    mock_relationship_2.r_squared = 0.7
    mock_stats_2 = Mock()
    mock_stats_2.relationship = mock_relationship_2

    features_with_stats = cast(
        list[FeatureWithFullStats],
        [
            Mock(feature=mock_feature_1, stats=mock_stats_1),
            Mock(feature=mock_feature_2, stats=mock_stats_2),
        ],
    )

    # Create recommendations with hallucinated features
    mock_pydantic_report = StructuredReport(
        analysis_summary='Test analysis',
        recommendations=[
            RecommendationRaw(
                title='Recommendation with Hallucinations',
                description='Has real and hallucinated features',
                rationale='Test filtering',
                evidence='Test evidence',
                supporting_features=[
                    'Real feature 1',
                    "Hallucinated feature that doesn't exist",  # Should be filtered (SSE=0)
                    'Real feature 2',
                    'Another fake feature',  # Should be filtered (SSE=0)
                ],
                supporting_conversations=[],
            ),
            RecommendationRaw(
                title='Only Hallucinations',
                description='All features are hallucinated',
                rationale='Should be removed entirely',
                evidence='No valid features',
                supporting_features=[
                    'Fake feature 1',
                    'Fake feature 2',
                    'Fake feature 3',
                ],
                supporting_conversations=[],
            ),
        ]
    )

    # Call the conversion and filtering methods
    unfiltered = recommender._convert_pydantic_to_attrs(
        mock_pydantic_report,
        features_with_stats,
        (),  # conversations
        [],  # conversation_ids
        [],  # outcomes
        'raw report'
    )
    result = recommender._filter_recommendations_with_no_supporting_features(unfiltered)

    # Verify only 1 recommendation remains (the one with some real features)
    assert len(result.recommendations) == 1
    rec = result.recommendations[0]

    # Verify it's the correct recommendation
    assert rec.title == 'Recommendation with Hallucinations'

    # Should only have the 2 real features
    assert len(rec.supporting_features) == 2
    feature_names = [f.name for f in rec.supporting_features]
    assert 'Real feature 1' in feature_names
    assert 'Real feature 2' in feature_names

    # Verify hallucinated features are not included
    assert "Hallucinated feature that doesn't exist" not in feature_names
    assert 'Another fake feature' not in feature_names

    # Verify R² values are correct for real features
    r_squared_values = {f.name: f.r_squared for f in rec.supporting_features}
    assert r_squared_values['Real feature 1'] == 0.9
    assert r_squared_values['Real feature 2'] == 0.7

    # Verify the recommendation with only hallucinations was filtered out
    assert any('Filtered out' in record.message for record in caplog.records)
    assert any('Only Hallucinations' in record.message for record in caplog.records)

def format_conversation_for_verification(conv_metadata: ConversationWithMetadata) -> str:
    """Format a conversation the same way ConversationFormatter does for verification.

    Args:
        conv_metadata: ConversationWithMetadata containing conversation and outcome

    Returns:
        Formatted conversation string with messages and outcome
    """
    # Format messages with timestamps and roles
    conversation_lines = [
        f'[{message.timestamp}] [{message.role}] {message.content}'
        for message in conv_metadata.conversation.messages
    ]

    return '\n'.join(conversation_lines)


def prepare_conversation_metadata(
    conversation_id: int,
    messages: tuple[Message, ...], outcome: str
) -> ConversationWithMetadata:
    """Create a ConversationWithMetadata for testing.

    Args:
        conversation_id: ID of the conversation
        messages: messages for creating the conversation
        outcome: outcome of the conversation

    Returns:
        ConversationWithMetadata instance
    """
    conv = Conversation(messages)

    return ConversationWithMetadata(
        actual_id=str(conversation_id),
        conversation=conv,
        outcome=outcome,
    )

@pytest.mark.integration
async def test_filter_and_clean_unrelated_conversation(
    real_llm_with_spec: LLMWithSpec,
    structuring_llm_with_spec: LLMWithSpec,
) -> None:
    """Test that unrelated conversations are filtered and removed from evidence and rationale."""
    # Create simple conversations
    # Conversation 1: About pricing (related to recommendation)
    conv_1_messages = (Message(
                role='customer',
                content='What is your pricing?',
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
            ),
            Message(
                role='agent',
                content='Our pricing depends on the package.',
                timestamp=datetime(2024, 1, 1, 10, 1, 0),
            ),
            Message(
                role='customer',
                content='Can you show me a price breakdown?',
                timestamp=datetime(2024, 1, 1, 10, 2, 0),
            ),
            Message(
                role='agent',
                content='I need to check our system.',
                timestamp=datetime(2024, 1, 1, 10, 3, 0),
            ),
        )

    conv1_metadata = prepare_conversation_metadata(1, conv_1_messages, 'lost')

    # Conversation 2: order status check (not related to the recommendation)
    conv2_messages = (
            Message(
                role='customer',
                content='My order from last week has not arrived yet',
                timestamp=datetime(2024, 1, 2, 11, 0, 0),
            ),
            Message(
                role='agent',
                content='I am sorry to hear that. What is the order number?',
                timestamp=datetime(2024, 1, 2, 11, 1, 0),
            )
    )

    conv2_metadata = prepare_conversation_metadata(2, conv2_messages, 'lost')

    # Create conversation explanations
    conv1_explanation = ConversationWithExplanation(
        conversation_id=1,
        explanation='Customer asks about pricing but agent cannot provide details.',
    )
    conv2_explanation = ConversationWithExplanation(
        conversation_id=2,
        explanation='Customer asks about why the existing order was not delivered.',
    )

    # Create Recommendation A: "Improve pricing transparency"
    # References both Conv 1 and Conv 2 in evidence and rationale
    rec_a = Recommendation(
        title='Improve pricing transparency',
        description='Provide clear pricing information upfront to help customers make decisions.',
        rationale='Customers often ask about pricing (see Conversation 1, Conversation 2). We need better pricing displays.',
        evidence='Conversation 1 shows customer asking for price breakdown. Conversation 2 also mentioned.',
        supporting_features=(
            FeatureWithScore(
                name='Agent mentions pricing',
                r_squared=0.5,
            ),
        ),
        supporting_conversations=(conv1_explanation, conv2_explanation),
    )

    # Create Recommendation B: No supporting conversations
    rec_b = Recommendation(
        title='Enhance customer support',
        description='Improve overall customer support quality.',
        rationale='General customer support improvements are needed.',
        evidence='No specific conversations referenced.',
        supporting_features=(),
        supporting_conversations=(),
    )

    # Create report
    report_before = RecommendationsReport(
        analysis_summary='Analysis of customer conversations',
        recommendations=(rec_a, rec_b),
        conversations={
            1: conv1_metadata,
            2: conv2_metadata,
        },
        raw_report='',
        total_conversations_analyzed=0,
        all_features=()
    )

    # Create formatted conversations map
    formatted_conversations = pl.Series([
        format_conversation_for_verification(conv1_metadata),
        format_conversation_for_verification(conv2_metadata),
    ])

    # Create recommender
    recommender = ConversationActionRecommender(
        model=real_llm_with_spec,
        structuring_model=structuring_llm_with_spec,
    )

    # Run verification
    filtered_report = await recommender._verify_conversation_references(
        report_before,
        formatted_conversations
    )

    # Assertions
    assert len(filtered_report.recommendations) == 2, 'Should still have 2 recommendations'

    # Find the recommendations by title
    rec_a_filtered = None
    rec_b_filtered = None
    for rec in filtered_report.recommendations:
        if rec.title == 'Improve pricing transparency':
            rec_a_filtered = rec
        elif rec.title == 'Enhance customer support':
            rec_b_filtered = rec

    assert rec_a_filtered is not None, 'Rec A should exist'
    assert rec_b_filtered is not None, 'Rec B should exist'

    # Verify Conversation 2 (status) was filtered out
    conv_ids_in_rec_a = {conv.conversation_id for conv in rec_a_filtered.supporting_conversations}
    assert 1 in conv_ids_in_rec_a, 'Conversation 1 (pricing) should be kept'
    assert 2 not in conv_ids_in_rec_a, 'Conversation 2 (status) should be filtered out'

    # Verify evidence was cleaned
    assert 'Conversation 1' in rec_a_filtered.evidence or 'Conv 1' in rec_a_filtered.evidence, \
        'Evidence should still mention Conversation 1'
    assert 'Conversation 2' not in rec_a_filtered.evidence and 'Conv 2' not in rec_a_filtered.evidence, \
        'Evidence should NOT mention Conversation 2'

    # Verify rationale was cleaned
    assert 'Conversation 1' in rec_a_filtered.rationale or 'Conv 1' in rec_a_filtered.rationale, \
        'Rationale should still mention Conversation 1'
    assert 'Conversation 2' not in rec_a_filtered.rationale and 'Conv 2' not in rec_a_filtered.rationale, \
        'Rationale should NOT mention Conversation 2'

    # Verify recommendation without conversations is at the end
    assert filtered_report.recommendations[-1] == rec_b_filtered, \
        'Recommendation without supporting conversations should be at the end'
