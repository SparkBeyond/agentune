from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import ClassVar

import polars as pl
from attrs import define, frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.feature.gen.base import FeatureGenerator, GeneratedFeature
from agentune.analyze.feature.gen.insightful_text_generator.dedup.base import QueryDeduplicator
from agentune.analyze.feature.gen.insightful_text_generator.dedup.llm_based_deduplicator import (
    LLMBasedDeduplicator,
)
from agentune.analyze.feature.gen.insightful_text_generator.features import create_feature
from agentune.analyze.feature.gen.insightful_text_generator.formatting.base import DataFormatter
from agentune.analyze.feature.gen.insightful_text_generator.formatting.conversation import (
    ShortDateConversationFormatter,
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
    FailedColumn,
    SuccessfulColumn,
    execute_llm_caching_aware_columnar,
    parse_json_response_field,
)
from agentune.analyze.feature.problem import Classification, Problem, Regression
from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.analyze.join.conversation import ConversationJoinStrategy
from agentune.core import types
from agentune.core.dataset import Dataset
from agentune.core.llm import LLMContext, LLMSpec
from agentune.core.progress.base import stage_scope
from agentune.core.progress.util import execute_and_count
from agentune.core.sercontext import LLMWithSpec

logger = logging.getLogger(__name__)


@frozen
class EnrichedQueryResult:
    """Result of enriching a single query with LLM-generated data.
    
    The query and enriched_values are always paired together, even if enrichment failed.
    Failed enrichments have all None values in enriched_values.
    """
    query: Query
    enriched_values: list[str | None]

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
        min_queries_percentage: Minimum percentage of requested queries that must be generated. 
            Requested queries = num_features_per_round * num_actionable_rounds + num_creative_features.
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
    min_queries_percentage: float = 0.5

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
    
    def _get_formatter(self, conversation_strategy: ConversationJoinStrategy, problem: Problem, generation_mode: bool) -> DataFormatter:
        params_to_print = (problem.target_column,) if generation_mode else ()
        return ShortDateConversationFormatter(
            name=f'conversation_formatter_{conversation_strategy.name}',
            conversation_strategy=conversation_strategy,
            params_to_print=params_to_print,
            include_in_batch_id=generation_mode
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
        formatter = self._get_formatter(conversation_strategy, problem, generation_mode=True)
        prompt_template = CREATIVE_FEATURES_PROMPT if creative else ACTIONABLE_QUESTIONNAIRE_PROMPT
        return ConversationQueryGenerator(
            model=self.query_generator_model,
            sampler=sampler,
            max_sample_size=self.max_samples_for_generation,
            prompt_template=prompt_template,
            formatter=formatter
        )

    async def enrich_queries(self, queries: list[Query], enrichment_formatter: DataFormatter, 
                             input_data: Dataset, conn: DuckDBPyConnection) -> list[EnrichedQueryResult]:
        """Enrich queries with LLM-generated conversation data.
        
        Returns a list of EnrichedQueryResult for successfully enriched queries only.
        Queries that fail enrichment (all None values) are filtered out.
        The returned list may be shorter than the input queries list, or empty if all queries fail.
        """
        if not enrichment_formatter.description:
            raise ValueError('DataFormatter must have a description for ConversationQueryGenerator.')

        with stage_scope('Enrich queries', count=0) as enrich_queries_stage:
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

            # Set total to the number of cells to execute
            total_cells = len(queries) * len(formatted_examples)
            enrich_queries_stage.set_total(total_cells)

            # Execute LLM calls with caching-aware staging
            response_columns = await execute_llm_caching_aware_columnar(self.query_enrich_model, prompt_columns, enrich_queries_stage)

            # Parse responses and create EnrichedQueryResult for each query
            results: list[EnrichedQueryResult] = []
            for query, column_result in zip(queries, response_columns, strict=True):
                if isinstance(column_result, SuccessfulColumn):
                    enriched_values = [
                        parse_json_response_field(resp, PARSER_OUT_FIELD)
                        for resp in column_result.values
                    ]
                    results.append(EnrichedQueryResult(query=query, enriched_values=enriched_values))
                elif isinstance(column_result, FailedColumn):
                    # For failed columns, create None values with the same length as examples
                    logger.warning(f'Query "{query.name}" failed with error: {column_result.exception}')
                    enriched_values = [None] * len(formatted_examples)
                    results.append(EnrichedQueryResult(query=query, enriched_values=enriched_values))
                else:
                    raise TypeError(f'Unexpected column result type: {type(column_result)}')
            
            # Filter out queries where all values are None (complete failures)
            successful_results = [r for r in results if not all(v is None for v in r.enriched_values)]
            
            if len(successful_results) < len(results):
                failed_count = len(results) - len(successful_results)
                logger.warning(f'{failed_count} of {len(results)} queries failed enrichment and were filtered out')
            
            return successful_results

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

    async def determine_dtypes(self, enriched_results: list[EnrichedQueryResult]) -> list[Query]:
        """Determine the appropriate dtype for each enriched query result.
        Returns a partial list, only for queries where type detection succeeded.
        """
        with stage_scope('Type detection', count=0, total=len(enriched_results)) as type_detection_stage:
            # Use gather to batch all dtype determinations
            results = await asyncio.gather(*[
                execute_and_count(
                    self._determine_dtype(r.query, pl.Series(r.query.name, r.enriched_values)),
                    type_detection_stage
                )
                for r in enriched_results
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

            # Check that we generated enough queries
            max_queries = self.num_actionable_rounds * self.num_features_per_round + self.num_creative_features
            if len(query_batch) < self.min_queries_percentage * max_queries:
                logger.error(f'Generated only {len(query_batch)} queries, which is less than the minimum required {self.min_queries_percentage * 100:.1f}% of {max_queries} requested queries.')
                raise RuntimeError(f'Generated only {len(query_batch)} queries, which is less than the minimum required {self.min_queries_percentage * 100:.1f}% of {max_queries} requested queries.'
                                   ' Try lowering num_features_per_round or adjusting other feature-generation parameters.')

            # 2. Enrich the queries with additional conversation information
            sampler = self._get_sampler(problem)
            sampled_data = sampler.sample(filtered_feature_search, self.num_samples_for_enrichment, self.random_seed)
            enrichment_formatter = self._get_formatter(conversation_strategy, problem, generation_mode=False)
            enriched_results = await self.enrich_queries(query_batch, enrichment_formatter, sampled_data, conn)
            
            # enrich_queries filters out failed queries, so enriched_results may be empty
            if not enriched_results:
                continue

            # 3. Determine the data types for the enriched queries
            updated_queries = await self.determine_dtypes(enriched_results)

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
        with stage_scope('Generate queries', 0) as queries_stage:
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
                queries_stage.increment_count(len(gen_queries))
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
                queries_stage.increment_count(len(creative_queries))
            # Final deduplication on all queries from both phases
            deduplicator = self._get_deduplicator()
            unique_queries = await deduplicator.deduplicate(queries)
            logger.debug(f'Deduplicated to {len(unique_queries)} unique queries after all generations')
            return unique_queries
