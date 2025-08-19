"""Schema definitions for the insightful text generator module.

This module contains the core data structures used throughout the query generation pipeline.
"""

from attrs import frozen

from agentune.analyze.core.types import Dtype


@frozen
class Query:
    """A single feature query"""
    name: str  # Unique identifier for the query
    query_text: str  # The actual query text to be asked
    return_type: Dtype  # Expected return type of the query (e.g. string, int, etc.)
    
