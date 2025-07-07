"""Module for Retrieval-Augmented Generation (RAG) components.

This module provides tools for creating and managing vector stores from conversation data,
which can then be used by RAG-enabled participants in the conversation simulator.
"""

from .indexing_and_retrieval_utils import (
    conversations_to_langchain_documents,
    get_similar_examples_for_next_message_role
)

__all__ = [
    "conversations_to_langchain_documents",
    "get_similar_examples_for_next_message_role",
]
