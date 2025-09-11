import asyncio
import logging
from collections.abc import AsyncIterator

import polars as pl
from attrs import define
from duckdb import DuckDBPyConnection

from agentune.analyze.core import types
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.feature.base import Feature
from agentune.analyze.feature.gen.base import FeatureGenerator, GeneratedFeature
from agentune.analyze.feature.gen.insightful_text_generator.dedup.base import (
    QueryDeduplicator,
    SimpleDeduplicator,
)
from agentune.analyze.feature.gen.insightful_text_generator.formatting.base import (
    ConversationFormatter,
)
from agentune.analyze.feature.gen.insightful_text_generator.prompts import (
    create_enrich_conversation_prompt,
)
from agentune.analyze.feature.gen.insightful_text_generator.query_generator import (
    ConversationQueryGenerator,
)
from agentune.analyze.feature.gen.insightful_text_generator.sampling.base import (
    DataSampler,
    RandomSampler,
)
from agentune.analyze.feature.gen.insightful_text_generator.schema import Query
from agentune.analyze.feature.gen.insightful_text_generator.type_detector import (
    cast_to_categorical,
    decide_dtype,
)
from agentune.analyze.feature.gen.insightful_text_generator.util import (
    execute_llm_caching_aware_columnar,
    parse_json_response_field,
)
from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.analyze.join.conversation import ConversationJoinStrategy

logger = logging.getLogger(__name__)


@define
class ConversationQueryFeatureGenerator[F: Feature](FeatureGenerator):
    # LLM and generation settings
    query_generator_model: LLMWithSpec
    num_samples_for_generation: int
    num_features_to_generate: int
    
    field_descriptions: str  # Description of available fields
    what_is_an_instance: str  # What each data point represents
    instance_description: str  # Full description for prompts
    target_value: str  # The target value we want to characterize

    query_enrich_model: LLMWithSpec
    num_samples_for_enrichment: int
    random_seed: int | None = None
    max_categorical: int = 9  # Max unique values for a categorical field
    max_empty_percentage: float = 0.5  # Max percentage of empty/None values allowed
    
    def _get_sampler(self, target_field: Field) -> DataSampler:
        # TODO: Implement logic to choose the appropriate sampler based on target field type
        if target_field.dtype.is_numeric():
            return RandomSampler()  # Replace with appropriate numeric sampler
        return RandomSampler()
    
    def _get_deduplicator(self) -> QueryDeduplicator:
        # TODO: upgrade to a more sophisticated deduplicator
        return SimpleDeduplicator()

    def find_conversation_strategies(self, join_strategies: TablesWithJoinStrategies) -> list[ConversationJoinStrategy]:
        return [
            strategy
            for table_with_strategies in join_strategies
            for strategy in table_with_strategies
            if isinstance(strategy, ConversationJoinStrategy)
        ]

    def create_query_generator(self, conversation_strategy: ConversationJoinStrategy, target_field: Field) -> ConversationQueryGenerator:
        """Create a ConversationQueryGenerator for the given conversation strategy."""
        sampler = self._get_sampler(target_field)
        deduplicator = self._get_deduplicator()
        return ConversationQueryGenerator(
            model=self.query_generator_model,
            sampler=sampler,
            sample_size=self.num_samples_for_generation,
            deduplicator=deduplicator,
            num_features_to_generate=self.num_features_to_generate,
            formatter=ConversationFormatter(
                name=f'conversation_formatter_{conversation_strategy.table.name}',
                conversation_strategy=conversation_strategy,
                params=Schema(cols=(target_field,))
            ),
            target_field=target_field,
            field_descriptions=self.field_descriptions,
            what_is_an_instance=self.what_is_an_instance,
            instance_description=self.instance_description,
            target_value=self.target_value
        )

    async def enrich_queries(self, queries: list[Query], enrichment_formatter: ConversationFormatter,
                             input_data: Dataset, conn: DuckDBPyConnection) -> pl.DataFrame:
        """Enrich a subset of queries with additional conversation information using parallel LLM calls.
        Returns a DataFrame containing the enriched query results
        """
        # Format the sampled data for enrichment
        formatted_examples = await enrichment_formatter.aformat_batch(input_data, conn)

        # Generate prompts for enrichment (columnar structure)
        prompt_columns = [
            [create_enrich_conversation_prompt(
                instance_description=self.instance_description,
                queries_str=f'{query.name}: {query.query_text}',
                instance=row
            ) for row in formatted_examples]
            for query in queries
        ]
        
        # Execute LLM calls with caching-aware staging
        response_columns = await execute_llm_caching_aware_columnar(self.query_enrich_model, prompt_columns)
        
        # Parse responses (already in optimal columnar structure)
        parsed_columns = [
            [parse_json_response_field(resp, 'response') for resp in column]
            for column in response_columns
        ]
        
        # Create DataFrame directly from columnar structure
        enriched_df_data = {
            query.name: column_data
            for query, column_data in zip(queries, parsed_columns, strict=False)
        }
        enriched_df = pl.DataFrame(enriched_df_data)
        return enriched_df

    async def _determine_dtype(self, query: Query, series_data: pl.Series) -> Query | None:
        """Determine the appropriate dtype for a query based on the series data.
        if no suitable dtype is found, return None.
        """
        # Check for empty rows (None or empty string)
        total_rows = len(series_data)
        if total_rows == 0:
            logger.warning(f'Query "{query.name}" has no data, skipping')
            return None
        
        empty_count = series_data.null_count() + (series_data == '').sum()
        empty_percentage = empty_count / total_rows
        
        if empty_percentage > self.max_empty_percentage:
            logger.warning(f'Query "{query.name}" has {empty_percentage:.2%} empty values (>{self.max_empty_percentage:.2%}), skipping')
            return None
        
        # Determine the dtype
        dtype = decide_dtype(query, series_data, self.max_categorical)
        # if dtype is string, try to cast to categorical
        if dtype == types.string:
            try:
                updated_query = await cast_to_categorical(
                    query,
                    series_data,
                    self.max_categorical,
                    self.query_generator_model
                )
                # Update the query and dtype
                if not isinstance(updated_query.return_type, types.EnumDtype):
                    raise TypeError('cast_to_categorical should return an EnumDtype')  # noqa: TRY301
                return updated_query
            except (ValueError, TypeError, AssertionError, RuntimeError) as e:
                logger.warning(f'Failed to cast query "{query.name}" to categorical, skipping: {e}')
                return None
        if not ((dtype in [types.boolean, types.int32, types.float64]) or isinstance(dtype, types.EnumDtype)):
            raise ValueError(f'Invalid dtype: {dtype}')
        return Query(name=query.name,
                     query_text=query.query_text,
                     return_type=dtype)

    async def determine_dtypes(self, queries: list[Query], enriched_output: pl.DataFrame) -> list[Query]:
        """Determine the appropriate dtype for each query based on the enriched output data.
        Returns a partial list, only for columns where type detection succeeded.
        """
        # Use gather to batch all dtype determinations
        results = await asyncio.gather(*[
            self._determine_dtype(q, enriched_output[q.name])
            for q in queries
        ])
        
        # Filter out None results
        return [query for query in results if query is not None]

    def create_features_from_queries(self, queries: list[Query], enrichment_formatter: ConversationFormatter,  # noqa: ARG002
                                     target_field: Field, conversation_strategy: ConversationJoinStrategy) -> list[F]:  # noqa: ARG002
        # TODO: update enriched output type
        return []  # Implement logic to create Features from the enriched queries

    async def agenerate(self, feature_search: Dataset, target_column: str, join_strategies: TablesWithJoinStrategies,
                        conn: DuckDBPyConnection) -> AsyncIterator[GeneratedFeature[F]]:
        target_field = feature_search.schema[target_column]
        conversation_strategies = self.find_conversation_strategies(join_strategies)

        for conversation_strategy in conversation_strategies:
            # 1. Create a query generator for the conversation
            query_generator = self.create_query_generator(conversation_strategy, target_field)

            # 2. Generate queries from the conversation data
            query_batch = await query_generator.agenerate_queries(feature_search, conn, self.random_seed)

            # 3. Enrich the queries with additional conversation information
            sampler = self._get_sampler(feature_search.schema[target_column])
            sampled_data = sampler.sample(feature_search, self.num_samples_for_enrichment, self.random_seed)
            enrichment_formatter = ConversationFormatter(
                name=f'enrichment_formatter_{conversation_strategy.table.name}',
                conversation_strategy=conversation_strategy,
                params=Schema(cols=())
            )
            enriched_output = await self.enrich_queries(query_batch, enrichment_formatter, sampled_data, conn)

            # 4. Determine the data types for the enriched queries
            updated_queries = await self.determine_dtypes(query_batch, enriched_output)

            # 5. Create Features from the enriched queries
            features = self.create_features_from_queries(
                queries=updated_queries,
                enrichment_formatter=enrichment_formatter,
                target_field=target_field,
                conversation_strategy=conversation_strategy
            )

            # Yield features one by one
            for feature in features:
                yield GeneratedFeature(feature, False)

