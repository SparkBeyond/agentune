"""Action recommendation module.

This module provides tools for generating actionable recommendations
based on features and their relationship to target variables.
"""

from agentune.improve.recommend.action_recommender import (
    ConversationActionRecommender,
    FeatureWithScore,
    Recommendation,
    RecommendationsReport,
)
from agentune.improve.recommend.base import ActionRecommender

__all__ = [
    'ActionRecommender',
    'ConversationActionRecommender',
    'FeatureWithScore',
    'Recommendation',
    'RecommendationsReport',
]
