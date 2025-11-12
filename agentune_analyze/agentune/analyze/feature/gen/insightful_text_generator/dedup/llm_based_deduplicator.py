
import logging

import attrs
from pydantic import BaseModel, Field

from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.feature.gen.insightful_text_generator.schema import Query

from .base import QueryDeduplicator

_logger = logging.getLogger(__name__)

class DedupQueryResponse(BaseModel):
    queries_to_keep: list[str] = Field(..., description='queries to keep')
    queries_to_remove: list[str] = Field(..., description='queries to remove')
    removal_mapping: dict[str, str] = Field(...,description='mapping from each removed query to the kept query it is most similar to')

@attrs.define
class LLMBasedDeduplicator(QueryDeduplicator):
    """Class for query deduplication using LLM support."""
    llm_with_spec: LLMWithSpec

    async def deduplicate(self, query_list: list[Query]) -> list[Query]:
        """Remove duplicate or highly similar queries from the list."""
        # Remove identical queries
        seen_query_texts = set()
        unique_full_query_set = []

        for query in query_list:
            query_text_lower = query.query_text.lower().strip()
            if query_text_lower not in seen_query_texts:
                seen_query_texts.add(query_text_lower)
                unique_full_query_set.append(query)

        # Get original query texts (not stripped/lowercased)
        original_seen_query_texts = [q.query_text for q in unique_full_query_set]

        # Format as a clean numbered list instead of raw Python list
        queries_numbered_list = '\n'.join([f'{i + 1}. {query}' for i, query in enumerate(original_seen_query_texts)])

        # Map full query to query text
        query_map = {q.query_text: q for q in unique_full_query_set}

        prompt = \
            f'''
            You are a helpful assistant that removes duplicate or almost duplicate questions. 
            Please differentiate between queries also by the answer type (Boolean, Number, String, Date, etc).

            Return ONLY valid JSON in the exact format:

            {{
            "queries_to_keep": [ "question1", "question2", ... ],
            "queries_to_remove": [ "dup1", "dup2", ... ]
            "removal_mapping": {{ "dup1": "question1", "dup2": "question2", ... }}
            }}

            No explanations, no numbering, no extra text.  

            Here is the list of queries to deduplicate:
        
            {queries_numbered_list}
            '''

        llm_with_structured_output = self.llm_with_spec.llm.as_structured_llm(DedupQueryResponse)
        completion = await llm_with_structured_output.acomplete(prompt)
        structured_answer: DedupQueryResponse = completion.raw

        # Validate that all kept queries were in the original list
        invalid_kept_queries = [q for q in structured_answer.queries_to_keep if q not in original_seen_query_texts]

        if invalid_kept_queries:
            _logger.warning(
                f'''Warning: LLM returned queries to keep that weren't in original list: {invalid_kept_queries}, Returning original unique query set without deduplication''')
            return unique_full_query_set

        # Map back to full Query objects
        full_queries_to_keep = [query_map[q] for q in structured_answer.queries_to_keep if q in query_map]

        return full_queries_to_keep

