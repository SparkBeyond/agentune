from collections.abc import AsyncIterator
from typing import Any

from attrs import define
from duckdb import DuckDBPyConnection

from agentune.analyze.context.base import TablesWithContextDefinitions
from agentune.analyze.context.conversation import ConversationContext
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.feature.base import Feature
from agentune.analyze.feature.gen.base import FeatureGenerator
from agentune.analyze.feature.gen.insightful_text_generator.dedup.base import (
    QueryDeduplicator,
    SimpleDeduplicator,
)
from agentune.analyze.feature.gen.insightful_text_generator.formatting.base import (
    ConversationFormatter,
)
from agentune.analyze.feature.gen.insightful_text_generator.query_generator import (
    ConversationQueryGenerator,
)
from agentune.analyze.feature.gen.insightful_text_generator.sampling.base import (
    DataSampler,
    RandomSampler,
)
from agentune.analyze.feature.gen.insightful_text_generator.schema import Query


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
    
    def _get_sampler(self, target_field: Field) -> DataSampler:
        # TODO: Implement logic to choose the appropriate sampler based on target field type
        if target_field.dtype.is_numeric():
            return RandomSampler()  # Replace with appropriate numeric sampler
        return RandomSampler()
    
    def _get_deduplicator(self) -> QueryDeduplicator:
        # TODO: upgrade to a more sophisticated deduplicator
        return SimpleDeduplicator()

    def find_conversation_contexts(self, contexts: TablesWithContextDefinitions) -> list[ConversationContext]:
        """Find the ConversationContext in the provided contexts."""
        return [
            context_def
            for table_with_context in contexts
            for context_def in table_with_context
            if isinstance(context_def, ConversationContext)
        ]

    def create_query_generator(self, conversation_context: ConversationContext, target_field: Field) -> ConversationQueryGenerator:
        """Create a ConversationQueryGenerator for the given conversation context."""
        sampler = self._get_sampler(target_field)
        deduplicator = self._get_deduplicator()
        return ConversationQueryGenerator(
            model=self.query_generator_model,
            sampler=sampler,
            sample_size=self.num_samples_for_generation,
            deduplicator=deduplicator,
            num_features_to_generate=self.num_features_to_generate,
            formatter=ConversationFormatter(
                name=f'conversation_formatter_{conversation_context.table.name}',
                conversation_context=conversation_context,
                params=Schema(cols=(target_field,))
            ),
            target_field=target_field,
            field_descriptions=self.field_descriptions,
            what_is_an_instance=self.what_is_an_instance,
            instance_description=self.instance_description,
            target_value=self.target_value
        )
    
    async def _enrich_queries(self, queries: list[Query], enrichment_formatter: ConversationFormatter,  # noqa: ARG002
                              input_data: Dataset, conn: DuckDBPyConnection) -> Any:  # noqa: ARG002
        # TODO: decide on output type and implement
        return []  # Implement logic to enrich queries with additional context information

    def create_features_from_queries(self, queries: list[Query], enrichment_formatter: ConversationFormatter,  # noqa: ARG002
                                     enriched_output: Any, target_field: Field,  # noqa: ARG002
                                     conversation_context: ConversationContext) -> list[F]:  # noqa: ARG002
        # TODO: update enriched output type
        return []  # Implement logic to create Features from the enriched queries

    async def agenerate(self, feature_search: Dataset, target_column: str, contexts: TablesWithContextDefinitions,
                        conn: DuckDBPyConnection) -> AsyncIterator[F]:
        target_field = feature_search.schema[target_column]
        conversation_contexts = self.find_conversation_contexts(contexts)

        for conversation_context in conversation_contexts:
            # 1. Create a query generator for the conversation context
            query_generator = self.create_query_generator(conversation_context, target_field)

            # 2. Generate queries from the conversation data
            query_batch = await query_generator.agenerate_queries(feature_search, conn, self.random_seed)

            # 3. Enrich the queries with additional context information
            enrichment_formatter = ConversationFormatter(
                name=f'enrichment_formatter_{conversation_context.table.name}',
                conversation_context=conversation_context,
                params=Schema(cols=())
            )

            enriched_output = await self._enrich_queries(query_batch, enrichment_formatter, feature_search, conn)

            # 4. Create Features from the enriched queries
            features = self.create_features_from_queries(
                queries=query_batch,
                enrichment_formatter=enrichment_formatter,
                enriched_output=enriched_output,
                target_field=target_field,
                conversation_context=conversation_context
            )

            # Yield features one by one
            for feature in features:
                yield feature

