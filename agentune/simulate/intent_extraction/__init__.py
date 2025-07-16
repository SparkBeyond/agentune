"""Intent extraction functionality for conversation analysis.

This module provides tools for extracting intents from conversations, helping to
understand the goals and purposes behind customer and agent messages.
"""

from .base import IntentExtractor
from .zeroshot import ZeroshotIntentExtractor

__all__ = [
    "IntentExtractor",
    "ZeroshotIntentExtractor",
]
