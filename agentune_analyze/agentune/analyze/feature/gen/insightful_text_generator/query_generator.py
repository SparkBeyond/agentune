"""Base interfaces and data structures for query generation.

This module defines the core abstractions for the Query Generation Pipeline,
following the Insightful Text Features architecture.
"""

import logging
from abc import ABC, abstractmethod

from attrs import define
from duckdb import DuckDBPyConnection

from agentune.analyze.core import types
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.feature.gen.insightful_text_generator.formatting.base import (
    DataFormatter,
)
from agentune.analyze.feature.gen.insightful_text_generator.prompts import (
    questionnaire_prompt_context,
)
from agentune.analyze.feature.gen.insightful_text_generator.sampling.base import (
    DataSampler,
)
from agentune.analyze.feature.gen.insightful_text_generator.schema import Query
from agentune.analyze.feature.gen.insightful_text_generator.util import (
    achat_raw,
    estimate_tokens,
    extract_json_from_response,
    get_max_input_context_window,
)
from agentune.analyze.feature.problem import Problem

logger = logging.getLogger(__name__)

# Minimum number of examples required for query generation
MIN_SAMPLES = 5


@define
class QueryGenerator(ABC):
    """Abstract base class for generating feature queries from conversation data."""
        
    # LLM and generation settings
    model: LLMWithSpec

    sampler: DataSampler
    max_sample_size: int
    prompt_template: str
    
    @abstractmethod
    async def generate_prompt(self, sampled_data: Dataset, problem: Problem, num_features: int,
                              existing_queries: list[Query] | None, conn: DuckDBPyConnection) -> str:
        """Convert sampled data into LLM prompt."""
        ...

    async def parse_and_validate_queries(self, raw_query: str) -> list[Query]:
        """Parse raw query text into structured Query objects and validate them."""
        try:
            queries = []
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

    async def agenerate_queries(self, input_data: Dataset, problem: Problem, num_features: int, conn: DuckDBPyConnection,
                                random_seed: int | None, existing_queries: list[Query] | None = None) -> list[Query]:
        """Generate a batch of feature queries from input conversation data."""
        # 1. Sample maximum amount of data initially
        sampled_data = self.sampler.sample(input_data, self.max_sample_size, random_seed=random_seed)

        # 2. Generate prompt
        prompt = await self.generate_prompt(sampled_data, problem, num_features, existing_queries, conn)

        # 3. Call the LLM with the prompts to generate raw query text
        raw_response = await achat_raw(self.model, prompt)

        # 4. Parse and validate the generated queries
        queries = await self.parse_and_validate_queries(raw_response)

        # 5. Return as batch
        return queries


@define
class ConversationQueryGenerator(QueryGenerator):
    """Concrete implementation for generating feature queries from conversation data."""
    formatter: DataFormatter

    async def generate_prompt(self, sampled_data: Dataset, problem: Problem, num_features: int,
                              existing_queries: list[Query] | None, conn: DuckDBPyConnection) -> str:
        """Convert sampled conversation data into parameters for the prompt template.
        
        Args:
            sampled_data: Sampled dataset for examples
            problem: Problem definition
            existing_queries: Optional list of existing queries to avoid duplication
            conn: Database connection
        """
        if not self.formatter.description:
            raise ValueError('DataFormatter must have a description for ConversationQueryGenerator.')
        # Format all the examples
        formatted_examples = await self.formatter.aformat_batch(sampled_data, conn)
        # Get token limits
        max_context_window = get_max_input_context_window(self.model)
        # Find the maximum number of examples that fit within token limit
        num_examples = len(formatted_examples)
        while num_examples >= MIN_SAMPLES:
            # Try with current number of examples
            current_examples = formatted_examples.to_list()[:num_examples]
            
            # Generate the prompt
            examples_str = '\n\n'.join(current_examples)
            prompt_params = questionnaire_prompt_context(
                examples=examples_str,
                problem=problem,
                instance_type='conversation',
                instance_description=self.formatter.description,
                n_queries=str(num_features),
                existing_queries=existing_queries if existing_queries is not None else []
            )
            prompt = self.prompt_template.format(**prompt_params)

            # Estimate tokens
            current_tokens = estimate_tokens(prompt, self.model)

            if current_tokens <= max_context_window:
                logger.info(
                    f'Fitted prompt with {num_examples} out of {len(formatted_examples)} examples, '
                    f'using {current_tokens} tokens which is < token limit ({max_context_window} tokens)'
                )
                return prompt
            
            # Remove one example and try again
            num_examples -= 1
        # We couldn't fit even with minimum samples
        raise ValueError(f'Cannot fit the minimal amount of needed sample conversations ({MIN_SAMPLES}) within the LLM token limit ({max_context_window}). Try providing shorter conversations as input.')

