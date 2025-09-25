import asyncio
import logging
from collections.abc import AsyncIterator

import polars as pl
from attrs import define
from duckdb import DuckDBPyConnection

from agentune.analyze.core import types
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.feature.gen.base import FeatureGenerator, GeneratedFeature
from agentune.analyze.feature.gen.insightful_text_generator.dedup.base import (
    QueryDeduplicator,
    SimpleDeduplicator,
)
from agentune.analyze.feature.gen.insightful_text_generator.features import (
    create_feature,
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
from agentune.analyze.feature.gen.insightful_text_generator.sampling.samplers import (
    BalancedClassSampler,
    ProportionalNumericSampler,
)
from agentune.analyze.feature.gen.insightful_text_generator.schema import PARSER_OUT_FIELD, Query
from agentune.analyze.feature.gen.insightful_text_generator.type_detector import (
    cast_to_categorical,
    decide_dtype,
)
from agentune.analyze.feature.gen.insightful_text_generator.util import (
    execute_llm_caching_aware_columnar,
    parse_json_response_field,
)
from agentune.analyze.feature.problem import Classification, Problem, Regression
from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.analyze.join.conversation import ConversationJoinStrategy

logger = logging.getLogger(__name__)


@define
class ConversationQueryFeatureGenerator(FeatureGenerator):
    # LLM and generation settings
    query_generator_model: LLMWithSpec
    num_samples_for_generation: int
    num_features_to_generate: int
    
    query_enrich_model: LLMWithSpec
    num_samples_for_enrichment: int
    random_seed: int | None = None
    max_categorical: int = 9  # Max unique values for a categorical field
    max_empty_percentage: float = 0.5  # Max percentage of empty/None values allowed
    
    def _get_sampler(self, problem: Problem) -> DataSampler:
        if problem.target_kind == Classification:
            return BalancedClassSampler(target_field=problem.target_column)
        if problem.target_kind == Regression:
            return ProportionalNumericSampler(target_field=problem.target_column, num_bins=3)
        return RandomSampler()
    
    def _get_deduplicator(self) -> QueryDeduplicator:
        # TODO: upgrade to a more sophisticated deduplicator
        return SimpleDeduplicator()
    
    def _get_formatter(self, conversation_strategy: ConversationJoinStrategy, problem: Problem, include_target: bool) -> ConversationFormatter:
        params_to_print = (problem.target_column,) if include_target else ()
        return ConversationFormatter(
            name=f'conversation_formatter_{conversation_strategy.name}',
            conversation_strategy=conversation_strategy,
            params_to_print=params_to_print
        )

    def find_conversation_strategies(self, join_strategies: TablesWithJoinStrategies) -> list[ConversationJoinStrategy]:
        return [
            strategy
            for table_with_strategies in join_strategies
            for strategy in table_with_strategies
            if isinstance(strategy, ConversationJoinStrategy)
        ]

    def create_query_generator(self, conversation_strategy: ConversationJoinStrategy, problem: Problem) -> ConversationQueryGenerator:
        """Create a ConversationQueryGenerator for the given conversation strategy."""
        sampler = self._get_sampler(problem)
        deduplicator = self._get_deduplicator()
        formatter = self._get_formatter(conversation_strategy, problem, include_target=True)
        return ConversationQueryGenerator(
            model=self.query_generator_model,
            sampler=sampler,
            sample_size=self.num_samples_for_generation,
            deduplicator=deduplicator,
            num_features_to_generate=self.num_features_to_generate,
            formatter=formatter
        )

    async def enrich_queries(self, queries: list[Query], enrichment_formatter: ConversationFormatter, 
                             input_data: Dataset, conn: DuckDBPyConnection) -> pl.DataFrame:
        """Enrich a subset of queries with additional conversation information using parallel LLM calls.
        Returns a DataFrame containing the enriched query results
        """
        if not enrichment_formatter.description:
            raise ValueError('DataFormatter must have a description for ConversationQueryGenerator.')
        # Format the sampled data for enrichment
        formatted_examples = await enrichment_formatter.aformat_batch(input_data, conn)

        # Generate prompts for enrichment (columnar structure)
        prompt_columns = [
            [create_enrich_conversation_prompt(
                instance_description=enrichment_formatter.description,
                queries_str=f'{query.name}: {query.query_text}',
                instance=row
            ) for row in formatted_examples]
            for query in queries
        ]
        
        # Execute LLM calls with caching-aware staging
        response_columns = await execute_llm_caching_aware_columnar(self.query_enrich_model, prompt_columns)
        
        # Parse responses (already in optimal columnar structure)
        parsed_columns = [
            [parse_json_response_field(resp, PARSER_OUT_FIELD) for resp in column]
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

    async def agenerate(self, feature_search: Dataset, problem: Problem, join_strategies: TablesWithJoinStrategies,
                        conn: DuckDBPyConnection) -> AsyncIterator[GeneratedFeature]:
        conversation_strategies = self.find_conversation_strategies(join_strategies)

        for conversation_strategy in conversation_strategies:
            # 1. Create a query generator for the conversation
            query_generator = self.create_query_generator(conversation_strategy, problem)

            # 2. Generate queries from the conversation data
            query_batch = await query_generator.agenerate_queries(feature_search, problem, conn, self.random_seed)

            # 3. Enrich the queries with additional conversation information
            sampler = self._get_sampler(problem)
            sampled_data = sampler.sample(feature_search, self.num_samples_for_enrichment, self.random_seed)
            enrichment_formatter = self._get_formatter(conversation_strategy, problem, include_target=False)
            enriched_output = await self.enrich_queries(query_batch, enrichment_formatter, sampled_data, conn)

            # 4. Determine the data types for the enriched queries
            updated_queries = await self.determine_dtypes(query_batch, enriched_output)

            # 5. Create Features from the enriched queries
            features = [create_feature(
                query=query,
                formatter=enrichment_formatter,
                model=self.query_enrich_model)
                for query in updated_queries]

            # Yield features one by one
            for feature in features:
                yield GeneratedFeature(feature, False)
