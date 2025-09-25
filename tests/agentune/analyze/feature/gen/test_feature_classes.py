"""Tests for insightful text feature classes."""

import httpx
import polars as pl
import pytest
from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbName, DuckdbTable
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.llm import LLMContext, LLMSpec
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.feature.gen.insightful_text_generator.features import (
    InsightfulBoolFeature,
    InsightfulCategoricalFeature,
    InsightfulFloatFeature,
    InsightfulIntFeature,
    create_feature,
)
from agentune.analyze.feature.gen.insightful_text_generator.formatting.base import (
    ConversationFormatter,
    DataFormatter,
)
from agentune.analyze.feature.gen.insightful_text_generator.schema import Query
from agentune.analyze.join.conversation import ConversationJoinStrategy


@frozen
class SimpleFormatter(DataFormatter):
    """Simple formatter that returns readable text."""
    params = Schema(cols=(
        Field(name='customer_id', dtype=types.int32),
        Field(name='message', dtype=types.string),))
    secondary_tables = ()
    join_strategies = ()
    description = 'customer message'

    async def aformat_batch(self, input: Dataset, conn: DuckDBPyConnection) -> pl.Series:  # noqa: ARG002
        """Format data as readable text."""
        results = []
        for row in input.data.iter_rows(named=True):
            formatted = f"Customer {row['customer_id']} says: '{row['message']}'"
            results.append(formatted)
        return pl.Series(results)


@pytest.fixture
def formatter() -> DataFormatter:
    """Simple formatter for tests."""
    return SimpleFormatter(name='simple_formatter')


@pytest.fixture
async def real_llm_with_spec(httpx_async_client: httpx.AsyncClient) -> LLMWithSpec:
    """Create a real LLM for testing."""
    llm_context = LLMContext(httpx_async_client)
    llm_spec = LLMSpec('openai', 'gpt-4o')
    llm_with_spec = LLMWithSpec(
        llm=llm_context.from_spec(llm_spec),
        spec=llm_spec
    )
    return llm_with_spec


class TestFeatureTypes:
    """Tests for each feature type using create_feature and real LLM evaluation."""

    def test_feature_objects_are_hashable(self, formatter: DataFormatter, real_llm_with_spec: LLMWithSpec) -> None:
        """Test that all feature objects are hashable and can be used in sets/dicts."""
        # Create different feature types
        bool_query = Query(
            name='is_positive',
            query_text='Is this customer message positive?',
            return_type=types.boolean
        )
        bool_feature = create_feature(bool_query, formatter, real_llm_with_spec)
        
        int_query = Query(
            name='word_count',
            query_text='How many words are in this customer message?',
            return_type=types.int32
        )
        int_feature = create_feature(int_query, formatter, real_llm_with_spec)
        
        float_query = Query(
            name='urgency_score',
            query_text='Rate the urgency of this customer message from 0.0 to 1.0',
            return_type=types.float64
        )
        float_feature = create_feature(float_query, formatter, real_llm_with_spec)
        
        categorical_query = Query(
            name='intent',
            query_text='What is the intent? Choose from: question, complaint, compliment, request',
            return_type=types.EnumDtype('question', 'complaint', 'compliment', 'request')
        )
        categorical_feature = create_feature(categorical_query, formatter, real_llm_with_spec)
        
        # Test that features can be hashed
        features = [bool_feature, int_feature, float_feature, categorical_feature]
        for feature in features:
            # Should not raise TypeError
            hash(feature)
        
        # Test that features can be used in sets
        feature_set = set(features)
        assert len(feature_set) == 4
        
        # Test that features can be used as dict keys
        feature_dict = {feature: feature.name for feature in features}
        assert len(feature_dict) == 4
        assert feature_dict[bool_feature] == 'is_positive'
        assert feature_dict[int_feature] == 'word_count'
        assert feature_dict[float_feature] == 'urgency_score'
        assert feature_dict[categorical_feature] == 'intent'

        # Additionally ensure the production ConversationFormatter and a feature using it are hashable.
        # Build a minimal conversations table schema
        conv_schema = Schema(
            cols=(
                Field('opportunity_id', types.string),
                Field('timestamp', types.timestamp),
                Field('type', types.string),  # role column can be str or enum
                Field('message', types.string),
            )
        )
        conv_table = DuckdbTable(
            name=DuckdbName(name='conversations', database='memory', schema='main'),
            schema=conv_schema,
        )

        # Conversation join strategy with named params
        join: ConversationJoinStrategy[str] = ConversationJoinStrategy(
            name='conv',
            table=conv_table,
            main_table_id_column=Field('opportunity_id', types.string),
            id_column=conv_schema['opportunity_id'],
            timestamp_column=conv_schema['timestamp'],
            role_column=conv_schema['type'],
            content_column=conv_schema['message'],
        )

        # Real formatter hashability
        conv_formatter = ConversationFormatter(name='conv_fmt', conversation_strategy=join)
        hash(conv_formatter)  # should not raise

        conv_query = Query(
            name='conv_word_count',
            query_text='How many messages are in the conversation?',
            return_type=types.int32,
        )
        conv_feature = create_feature(conv_query, conv_formatter, real_llm_with_spec)
        # Should be hashable and usable as a dict key
        hash(conv_feature)
        d = {conv_feature: conv_feature.name}
        assert d[conv_feature] == 'conv_word_count'

    @pytest.mark.integration
    async def test_bool_feature_sentiment(self, formatter: DataFormatter, real_llm_with_spec: LLMWithSpec, conn: DuckDBPyConnection) -> None:
        """Test boolean feature creation and evaluation for sentiment analysis."""
        query = Query(
            name='is_positive',
            query_text='Is this customer message positive?',
            return_type=types.boolean
        )
        
        feature = create_feature(query, formatter, real_llm_with_spec)
        
        # Test feature was created correctly
        assert isinstance(feature, InsightfulBoolFeature)
        assert feature.name == 'is_positive'
        
        # Test positive message
        result = await feature.aevaluate((1001, 'Thank you so much! This is perfect!'), conn)
        assert isinstance(result, bool)
        assert result is True
        
        # Test negative message
        result = await feature.aevaluate((1002, 'This is terrible! I hate it!'), conn)
        assert isinstance(result, bool)
        assert result is False

    @pytest.mark.integration
    async def test_int_feature_word_count(self, formatter: DataFormatter, real_llm_with_spec: LLMWithSpec, conn: DuckDBPyConnection) -> None:
        """Test integer feature creation and evaluation for word counting."""
        query = Query(
            name='word_count',
            query_text='How many words are in this customer message?',
            return_type=types.int32
        )
        
        feature = create_feature(query, formatter, real_llm_with_spec)
        
        # Test feature was created correctly
        assert isinstance(feature, InsightfulIntFeature)
        assert feature.name == 'word_count'
        
        # Test word counting
        result = await feature.aevaluate((1001, 'Can you help me please?'),  conn)
        assert isinstance(result, int)
        assert result == 5

    @pytest.mark.integration
    async def test_float_feature_urgency(self, formatter: DataFormatter, real_llm_with_spec: LLMWithSpec, conn: DuckDBPyConnection) -> None:
        """Test float feature creation and evaluation for urgency scoring."""
        query = Query(
            name='urgency_score',
            query_text='Rate the urgency of this customer message from 0.0 (not urgent) to 1.0 (very urgent)',
            return_type=types.float64
        )
        
        feature = create_feature(query, formatter, real_llm_with_spec)
        
        # Test feature was created correctly
        assert isinstance(feature, InsightfulFloatFeature)
        assert feature.name == 'urgency_score'
        
        # Test high urgency
        result = await feature.aevaluate((1001, 'URGENT! Account locked, need immediate help!'), conn)
        assert isinstance(result, float)
        assert 0.7 <= result <= 1.0
        
        # Test low urgency
        result = await feature.aevaluate((1002, 'When convenient, could you help with this feature?'), conn)
        assert isinstance(result, float)
        assert 0.0 <= result <= 0.4

    @pytest.mark.integration
    async def test_categorical_feature_intent(self, formatter: DataFormatter, real_llm_with_spec: LLMWithSpec, conn: DuckDBPyConnection) -> None:
        """Test categorical feature creation and evaluation for intent classification."""
        categories = ['question', 'complaint', 'compliment', 'request']
        query = Query(
            name='intent',
            query_text='What is the intent of this customer message? Choose from: question, complaint, compliment, request',
            return_type=types.EnumDtype(*categories)
        )
        
        feature = create_feature(query, formatter, real_llm_with_spec)
        
        # Test feature was created correctly
        assert isinstance(feature, InsightfulCategoricalFeature)
        assert feature.name == 'intent'
        assert feature.categories == tuple(categories)
        
        # Test different intents
        result = await feature.aevaluate((1001, 'How do I reset my password?'), conn)
        assert result == 'question'

        result = await feature.aevaluate((1002, 'This service is terrible!'), conn)
        assert result == 'complaint'

        result = await feature.aevaluate((1003, 'Great job! Love this feature!'), conn)
        assert result == 'compliment'

        result = await feature.aevaluate((1004, 'Please update my billing address'), conn)
        assert result == 'request'
