"""Module for Retrieval-Augmented Generation (RAG) components.

This module provides tools for creating and managing vector stores from conversation data,
which can then be used by RAG-enabled participants in the agentune simulate.
"""

from .indexing_and_retrieval import (
    conversations_to_langchain_documents,
    get_few_shot_examples,
    get_similar_finished_conversations
)

__all__ = [
    "conversations_to_langchain_documents",
    "get_few_shot_examples",
    "get_similar_finished_conversations",
]
