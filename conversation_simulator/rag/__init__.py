"""Module for Retrieval-Augmented Generation (RAG) components.

This module provides tools for creating and managing vector stores from conversation data,
which can then be used by RAG-enabled participants in the conversation simulator.
"""

from .processing import (
    create_vector_stores_from_conversations,
    _get_few_shot_examples,
)

__all__ = [
    "create_vector_stores_from_conversations",
    "_get_few_shot_examples",
]
