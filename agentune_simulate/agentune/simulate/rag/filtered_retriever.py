"""
Polymorphic vector store searcher with efficient native filtering and adaptive fallback.

This module provides a unified interface for searching vector stores with metadata filtering,
automatically choosing between native server-side filtering (when supported) and adaptive
client-side filtering fallback for unsupported stores.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class VectorStoreSearcher(ABC):
    """Abstract base class for vector store searchers with metadata filtering."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    @abstractmethod
    async def similarity_search_with_filter(
        self, query: str, k: int, filter_dict: dict[str, Any]
    ) -> list[tuple[Document, float]]:
        """Search with metadata filtering - native or fallback implementation."""
        pass

    @staticmethod
    def create(vector_store: VectorStore) -> "VectorStoreSearcher":
        """Factory method to create appropriate searcher for vector store type."""
        store_type = type(vector_store).__name__
        
        if store_type == "InMemoryVectorStore":
            return InMemorySearcher(vector_store)
        elif store_type in ["Chroma"]:
            return DictionaryFilterSearcher(vector_store)
        else:
            logger.info(f"Using fallback filtering for unsupported vector store: {store_type}")
            return FallbackSearcher(vector_store)


class InMemorySearcher(VectorStoreSearcher):
    """Searcher for InMemoryVectorStore using function-based filters."""

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

    async def similarity_search_with_filter(
        self, query: str, k: int, filter_dict: dict[str, Any]
    ) -> list[tuple[Document, float]]:
        base_oversample = 4
        max_total_fetch = min(k * 20, 1000)  # Absolute maximum: 1000 documents
        
        current_fetch = k * base_oversample
        
        logger.debug(f"FallbackSearcher: starting adaptive search for {k} results with filter {filter_dict}")
        
        while current_fetch <= max_total_fetch:
            fetch_size = min(current_fetch, max_total_fetch)
            
            try:
                raw_results = await self.vector_store.asimilarity_search_with_score(
                    query=query, k=fetch_size
                )
            except Exception as e:
                logger.error(f"FallbackSearcher: vector store search failed: {e}")
                return []
            
            total_fetched = len(raw_results)
            
            # Client-side filtering
            filtered = [
                (doc, score) for doc, score in raw_results 
                if all(doc.metadata.get(key) == value for key, value in filter_dict.items())
            ]
            
            filter_ratio = len(filtered) / len(raw_results) if raw_results else 0
            logger.debug(
                f"FallbackSearcher: fetched {len(raw_results)}, filtered to {len(filtered)} "
                f"(ratio: {filter_ratio:.2f}), needed: {k}"
            )
            
            # Success: we have enough results
            if len(filtered) >= k:
                logger.debug(f"FallbackSearcher: success, returning {k} results")
                return filtered[:k]
                
            # No more documents available or hit absolute limit
            if len(raw_results) < current_fetch or current_fetch >= max_total_fetch:
                logger.warning(
                    f"FallbackSearcher: reached limit, returning {len(filtered)} results "
                    f"(requested: {k}, fetched: {total_fetched}, max: {max_total_fetch})"
                )
                return filtered
                
            # Double the fetch size for next attempt (but respect absolute max)
            current_fetch = min(current_fetch * 2, max_total_fetch)
        
        logger.warning(f"FallbackSearcher: exhausted all attempts, returning {len(filtered)} results")
        return filtered