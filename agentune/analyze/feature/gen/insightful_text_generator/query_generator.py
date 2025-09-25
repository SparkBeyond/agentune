"""Base interfaces and data structures for query generation.

This module defines the core abstractions for the Query Generation Pipeline,
following the Insightful Text Features architecture.
"""

import asyncio
import logging
from abc import ABC, abstractmethod

from attrs import define
from duckdb import DuckDBPyConnection

from agentune.analyze.core import types
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.feature.gen.insightful_text_generator.dedup.base import (
    QueryDeduplicator,
)
from agentune.analyze.feature.gen.insightful_text_generator.formatting.base import (
    DataFormatter,
)
from agentune.analyze.feature.gen.insightful_text_generator.prompts import (
    create_questionnaire_prompt,
)
from agentune.analyze.feature.gen.insightful_text_generator.sampling.base import (
    DataSampler,
)
from agentune.analyze.feature.gen.insightful_text_generator.schema import Query
from agentune.analyze.feature.gen.insightful_text_generator.util import (
    achat_raw,
    extract_json_from_response,
)
from agentune.analyze.feature.problem import Problem

logger = logging.getLogger(__name__)


@define
class QueryGenerator(ABC):
    """Abstract base class for generating feature queries from conversation data."""
        
    # LLM and generation settings
    model: LLMWithSpec

    sampler: DataSampler
    sample_size: int
    deduplicator: QueryDeduplicator
    num_features_to_generate: int
    
    @abstractmethod
    async def generate_prompts(self, sampled_data: Dataset, problem: Problem, conn: DuckDBPyConnection) -> list[str]:
        """Convert sampled data into prompts for the LLM."""
        ...

    async def parse_and_validate_queries(self, raw_queries: list[str]) -> list[Query]:
        """Parse raw query text into structured Query objects and validate them."""
        try:

            queries = []
            for raw_query in raw_queries:
                # Extract JSON from the response
                queries_dict = extract_json_from_response(raw_query)

                for query_name, query_text in queries_dict.items():
                    if not query_text:
                        continue
                    
                    # The real return type will be determined later
                    return_type = types.string

                    query = Query(
                        name=query_name,
                        query_text=query_text,
                        return_type=return_type
                    )
                    queries.append(query)
                
            return queries
            
        except Exception:
            logger.exception('Failed to parse queries')
            return []

    async def deduplicate_queries(self, queries: list[Query]) -> list[Query]:
        """Remove duplicate or highly similar queries from the generated set."""
        return await self.deduplicator.deduplicate(queries)

    async def limit_queries(self, queries: list[Query]) -> list[Query]:
        """Limit the number of queries to the configured maximum.

        Default implementation takes first N queries. Child classes can override
        for more sophisticated selection logic (e.g., diversity-based selection).
        """
        return queries[:self.num_features_to_generate]

    async def agenerate_queries(self, input_data: Dataset, problem: Problem, conn: DuckDBPyConnection, random_seed: int | None) -> list[Query]:
        """Generate a batch of feature queries from input conversation data."""
        # 1. Sample representative data for examples
        sampled_data = self.sampler.sample(input_data, self.sample_size, random_seed=random_seed)

        # 2. Create prompts from the sampled data
        prompts = await self.generate_prompts(sampled_data, problem, conn)

        # 3. Call the LLM with the prompts to generate raw query text
        raw_response = await asyncio.gather(*[achat_raw(self.model, prompt) for prompt in prompts])

        # 4. Parse and validate the generated queries
        queries = await self.parse_and_validate_queries(raw_response)

        # 5. Remove duplicates
        distinct_queries = await self.deduplicate_queries(queries)

        # 6. Limit to desired number of features
        final_queries = await self.limit_queries(distinct_queries)

        # 7. Return as batch
        return final_queries


@define
class ConversationQueryGenerator(QueryGenerator):
    """Concrete implementation for generating feature queries from conversation data."""
    formatter: DataFormatter

    async def generate_prompts(self, sampled_data: Dataset, problem: Problem, conn: DuckDBPyConnection) -> list[str]:
        """Convert sampled conversation data into parameters for the prompt template."""
        if not self.formatter.description:
            raise ValueError('DataFormatter must have a description for ConversationQueryGenerator.')
        # Get formatted examples from the sampled data using the formatter column name
        formatted_examples = await self.formatter.aformat_batch(sampled_data, conn)

        examples_str = '\n\n'.join(formatted_examples.to_list())

        prompt = create_questionnaire_prompt(
            examples=examples_str,
            problem=problem,
            instance_type='conversation',
            instance_description=self.formatter.description,
            n_queries=str(self.num_features_to_generate)
        )
        return [prompt]


