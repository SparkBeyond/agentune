from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import ClassVar

import polars as pl
from attrs import define
from duckdb import DuckDBPyConnection

from agentune.analyze.core import types
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.llm import LLMContext, LLMSpec
from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.feature.gen.base import FeatureGenerator, GeneratedFeature
from agentune.analyze.feature.gen.insightful_text_generator.dedup.base import QueryDeduplicator
from agentune.analyze.feature.gen.insightful_text_generator.dedup.llm_based_deduplicator import (
    LLMBasedDeduplicator,
)
from agentune.analyze.feature.gen.insightful_text_generator.features import create_feature
from agentune.analyze.feature.gen.insightful_text_generator.formatting.base import (
    ConversationFormatter,
)
from agentune.analyze.feature.gen.insightful_text_generator.prompts import (
    ACTIONABLE_QUESTIONNAIRE_PROMPT,
    CREATIVE_FEATURES_PROMPT,
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

SEED_OFFSET = 17



@define
class ConversationQueryFeatureGenerator(FeatureGenerator):
    """A feature generator that creates insightful features from conversation data using LLM-based query generation.

    This generator works in multiple phases:
    1. Generates analytical queries about conversations using LLMs
    2. Enriches the queries with additional conversation context
    3. Determines appropriate data types for the generated features
    4. Creates and returns feature objects

    The generator supports two types of queries:
    - Actionable queries: Focus on practical, actionable insights from conversations
    - Creative queries: Focus on interesting, potentially valuable patterns in conversations

    Args:
        query_generator_model: LLM model used for generating queries about conversations
        query_enrich_model: LLM model used for enriching queries with conversation context
        num_samples_for_generation: Number of conversation samples used when generating queries
        num_samples_for_enrichment: Number of conversation samples used when enriching queries
        num_features_per_round: Number of features to generate in each actionable round
        num_actionable_rounds: Number of rounds to generate actionable features
        num_creative_features: Number of additional creative features to generate
        random_seed: Random seed for reproducible sampling
        max_categorical: Maximum number of unique values allowed for categorical features
        max_empty_percentage: Maximum percentage of empty/None values allowed in features
    """
    default_query_generation_model: ClassVar[LLMSpec] = LLMSpec('openai', 'o3')
    default_query_enrichment_model: ClassVar[LLMSpec] = LLMSpec('openai', 'gpt-4o-mini')

    @staticmethod
    def default(llm_context: LLMContext) -> ConversationQueryFeatureGenerator:
        return ConversationQueryFeatureGenerator(
            query_generator_model=LLMWithSpec(ConversationQueryFeatureGenerator.default_query_generation_model, llm_context.from_spec(ConversationQueryFeatureGenerator.default_query_generation_model)),
            query_enrich_model=LLMWithSpec(ConversationQueryFeatureGenerator.default_query_enrichment_model, llm_context.from_spec(ConversationQueryFeatureGenerator.default_query_enrichment_model)),
        )

    # LLM and generation settings
    query_generator_model: LLMWithSpec
    query_enrich_model: LLMWithSpec

    # Optional parameters with defaults
    max_samples_for_generation: int = 30
    num_samples_for_enrichment: int = 200
    num_features_per_round: int = 20
    num_actionable_rounds: int = 2
    num_creative_features: int = 20

    random_seed: int | None = 42
    max_categorical: int = 9  # Max unique values for a categorical field
    max_empty_percentage: float = 0.5  # Max percentage of empty/None values allowed
    
    def _get_sampler(self, problem: Problem) -> DataSampler:
        if problem.target_kind == Classification:
            return BalancedClassSampler(target_field=problem.target_column)
        if problem.target_kind == Regression:
            return ProportionalNumericSampler(target_field=problem.target_column, num_bins=3)
        return RandomSampler()
    
    def _get_deduplicator(self) -> QueryDeduplicator:
        return LLMBasedDeduplicator(llm_with_spec=self.query_generator_model)
    
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

    def create_query_generator(self, conversation_strategy: ConversationJoinStrategy, problem: Problem, creative: bool = False) -> ConversationQueryGenerator:
        """Create a ConversationQueryGenerator for the given conversation strategy."""
        sampler = self._get_sampler(problem)
        formatter = self._get_formatter(conversation_strategy, problem, include_target=True)
        prompt_template = CREATIVE_FEATURES_PROMPT if creative else ACTIONABLE_QUESTIONNAIRE_PROMPT
        return ConversationQueryGenerator(
            model=self.query_generator_model,
            sampler=sampler,
            max_sample_size=self.max_samples_for_generation,
            prompt_template=prompt_template,
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
        if no suitable dtype is found, cast to categorical.
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
            # filter feature_search to only rows where conversation exists
            existing_ids = conversation_strategy.ids_exist(feature_search, conn)
            filtered_feature_search = Dataset(schema=feature_search.schema, data=feature_search.data.filter(existing_ids))

            # 1. Generate queries
            query_batch = await self._generate_queries(conversation_strategy, filtered_feature_search, problem, conn)

            # 2. Enrich the queries with additional conversation information
            sampler = self._get_sampler(problem)
            sampled_data = sampler.sample(filtered_feature_search, self.num_samples_for_enrichment, self.random_seed)
            enrichment_formatter = self._get_formatter(conversation_strategy, problem, include_target=False)
            enriched_output = await self.enrich_queries(query_batch, enrichment_formatter, sampled_data, conn)

            # 3. Determine the data types for the enriched queries
            updated_queries = await self.determine_dtypes(query_batch, enriched_output)

            # 4. Create Features from the enriched queries
            features = [create_feature(
                query=query,
                formatter=enrichment_formatter,
                model=self.query_enrich_model)
                for query in updated_queries]

            # Yield features one by one
            for feature in features:
                yield GeneratedFeature(feature, False)

    async def _generate_queries(self, conversation_strategy: ConversationJoinStrategy, input_data: Dataset, problem: Problem, conn: DuckDBPyConnection) -> list[Query]:
        """Generate queries num_generations times, each generating num_features_per_generation queries,
        followed by generating num_juicy_features juicy features.
        Finally deduplicate all generated queries and return the unique set.
        """
        query_generator = self.create_query_generator(conversation_strategy, problem, creative=False)
        current_seed = self.random_seed
        queries: list[Query] = []
        for gen_idx in range(self.num_actionable_rounds):
            logger.debug(f'Starting generation {gen_idx + 1}/{self.num_actionable_rounds} for conversation strategy "{conversation_strategy.name}"')
            gen_queries = await query_generator.agenerate_queries(
                input_data,
                problem,
                self.num_features_per_round,
                conn,
                random_seed=current_seed,
                existing_queries=queries
            )
            logger.debug(f'Generated {len(gen_queries)} queries in generation {gen_idx + 1}/{self.num_actionable_rounds}')
            queries.extend(gen_queries)
            if current_seed is not None:
                current_seed += SEED_OFFSET  # Offset seed for next generation to sample different conversations

        creative_query_generator = self.create_query_generator(conversation_strategy, problem, creative=True)
        if self.num_creative_features > 0:
            logger.debug(f'Generating additional {self.num_creative_features} juicy features for conversation strategy "{conversation_strategy.name}"')
            creative_queries = await creative_query_generator.agenerate_queries(
                input_data,
                problem,
                self.num_creative_features,
                conn,
                random_seed=current_seed,
                existing_queries=queries
            )
            logger.debug(f'Generated {len(creative_queries)} creative queries')
            queries.extend(creative_queries)

        # Final deduplication on all queries from both phases
        deduplicator = self._get_deduplicator()
        unique_queries = await deduplicator.deduplicate(queries)
        logger.debug(f'Deduplicated to {len(unique_queries)} unique queries after all generations')
        return unique_queries
