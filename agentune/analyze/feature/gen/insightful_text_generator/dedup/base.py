"""Base classes for query deduplication.

This module defines the core interfaces for query deduplication operations.
"""

from abc import ABC, abstractmethod
from typing import override

import attrs

from agentune.analyze.feature.gen.insightful_text_generator.schema import Query


@attrs.define
class QueryDeduplicator(ABC):
    """Abstract base class for query deduplication strategies."""

    @abstractmethod
    async def deduplicate(self, query_list: list[Query]) -> list[Query]:
        """Remove duplicate or highly similar queries from the list."""
        ...


@attrs.define
class SimpleDeduplicator(QueryDeduplicator):
    """Simple string-based deduplication without sophisticated similarity checking.
    
    This deduplicator uses exact string matching (case-insensitive) to remove duplicates.
    """

    @override
    async def deduplicate(self, query_list: list[Query]) -> list[Query]:
        """Deduplicate using simple string matching."""
        seen_queries = set()
        unique_query_set = []
        
        for query in query_list:
            query_key = query.query_text.lower().strip()
            if query_key not in seen_queries:
                seen_queries.add(query_key)
                unique_query_set.append(query)
                
        return unique_query_set
    
