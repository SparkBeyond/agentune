"""
Polymorphic vector store searcher with efficient native filtering and adaptive fallback.

This module provides a unified interface for searching vector stores with metadata filtering,
automatically choosing between native server-side filtering (when supported) and adaptive
client-side filtering fallback for unsupported stores.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, override

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore

try:
    from langchain_chroma import Chroma
    _IS_CHROMA_AVAILABLE = True
except ImportError:
    _IS_CHROMA_AVAILABLE = False


logger = logging.getLogger(__name__)

_MAX_TOTAL_FETCH = 2000  # Absolute limit for number of documents to fetch in fallback search


class VectorStoreSearcher(ABC):
    """Abstract base class for vector store searchers with metadata filtering."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    @abstractmethod
    async def similarity_search_with_filter(
        self, query: str, k: int, filter_dict: dict[str, Any]
    ) -> list[tuple[Document, float]]:
        """Search with metadata filtering - native or fallback implementation.
        
        Args:
            query: Text query for similarity search
            k: Number of results to return
            filter_dict: Dictionary of metadata key-value pairs for filtering.
                        Documents are included only if ALL key-value pairs match
                        their metadata exactly. Example:
                        {"current_message_role": "agent", "has_next_message": False}
                        
        Returns:
            List of (Document, similarity_score) tuples matching the filter criteria
        """
        pass

    @staticmethod
    def create(vector_store: VectorStore) -> VectorStoreSearcher:
        """Factory method to create appropriate searcher for vector store type."""
        if isinstance(vector_store, InMemoryVectorStore):
            return InMemorySearcher(vector_store)
        elif _IS_CHROMA_AVAILABLE and isinstance(vector_store, Chroma):
            return DictionaryFilterSearcher(vector_store)
        else:
            logger.debug(f"Using fallback filtering for unsupported vector store: {type(vector_store).__name__}")
            return FallbackSearcher(vector_store)


class InMemorySearcher(VectorStoreSearcher):
    """Searcher for InMemoryVectorStore using function-based filters."""

    @override
    async def similarity_search_with_filter(
        self, query: str, k: int, filter_dict: dict[str, Any]
    ) -> list[tuple[Document, float]]:
        def filter_func(doc: Document) -> bool:
            return all(doc.metadata.get(key) == value for key, value in filter_dict.items())
        
        logger.debug(f"InMemorySearcher: searching with filter {filter_dict}")
        results: list[tuple[Document, float]] = await self.vector_store.asimilarity_search_with_score(
            query=query, k=k, filter=filter_func
        )
        logger.debug(f"InMemorySearcher: found {len(results)} results")
        return results


class DictionaryFilterSearcher(VectorStoreSearcher):
    """Searcher for vector stores that support dictionary-based filters (Chroma)."""

    @override
    async def similarity_search_with_filter(
        self, query: str, k: int, filter_dict: dict[str, Any]
    ) -> list[tuple[Document, float]]:
        logger.debug(f"DictionaryFilterSearcher: searching with filter {filter_dict}")
        results: list[tuple[Document, float]] = await self.vector_store.asimilarity_search_with_score(
            query=query, k=k, filter=filter_dict
        )
        logger.debug(f"DictionaryFilterSearcher: found {len(results)} results")
        return results


class FallbackSearcher(VectorStoreSearcher):
    """Client-side filtering fallback with adaptive sampling and absolute limits."""

    @staticmethod
    def _filter(
            results: list[tuple[Document, float]],
            filter_dict: dict[str, Any]
    ) -> list[tuple[Document, float]]:
        """Filter results based on metadata dictionary."""
        return [
            (doc, score) for doc, score in results
            if all(doc.metadata.get(key) == value for key, value in filter_dict.items())
        ]

    @staticmethod
    def _sort(
        results: list[tuple[Document, float]]
    ) -> list[tuple[Document, float]]:
        """Sort results by score in descending order."""
        return sorted(results, key=lambda x: x[1], reverse=True)

    @override
    async def similarity_search_with_filter(
        self, query: str, k: int, filter_dict: dict[str, Any]
    ) -> list[tuple[Document, float]]:
        first_fetch = min(k * 4, _MAX_TOTAL_FETCH)
        second_fetch = min(k * 40, _MAX_TOTAL_FETCH)

        raw_results = await self.vector_store.asimilarity_search_with_score(query=query, k=first_fetch)
        filtered_results = self._filter(raw_results, filter_dict)

        if len(raw_results) < first_fetch or len(filtered_results) >= k:
            return  self._sort(filtered_results)

        logger.debug(f"fetching second time with adaptive sampling: first fetch returned {len(raw_results)} results out of {first_fetch}, filtered {len(filtered_results)} results, requested {k}")

        raw_results = await self.vector_store.asimilarity_search_with_score(query=query, k=second_fetch)
        filtered_results = self._filter(raw_results, filter_dict)

        if len(raw_results) >= second_fetch and len(filtered_results) < k:
            logger.debug(f"FallbackSearcher: second fetch returned {len(raw_results)} results, after filtering {len(filtered_results)} results, requested {k}.")

        return self._sort(filtered_results)
