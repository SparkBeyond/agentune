"""Module for Retrieval-Augmented Generation (RAG) components.

This module provides tools for creating and managing vector stores from conversation data,
which can then be used by RAG-enabled participants in the conversation simulator.
"""

from .commons import (
    conversations_to_langchain_documents,
    get_few_shot_examples,
)

__all__ = [
    "conversations_to_langchain_documents",
    "get_few_shot_examples",
]
