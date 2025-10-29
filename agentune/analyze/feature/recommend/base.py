"""Base classes for action recommendations.

This module defines the core interface for generating actionable recommendations
based on features and their relationship to target variables.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from duckdb import DuckDBPyConnection

from agentune.analyze.core.dataset import Dataset
from agentune.analyze.feature.problem import Problem
from agentune.analyze.feature.stats.base import FeatureWithFullStats

if TYPE_CHECKING:
    from agentune.analyze.feature.recommend.action_recommender import RecommendationsReport


class ActionRecommender(ABC):
    """Abstract base class for action recommendation strategies.
    
    Recommenders generate actionable recommendations based on features and their
    relationship to the target variable. Different recommenders may use different
    approaches (LLM-based, rule-based, statistical, etc.).
    
    Each recommender returns its own report type (which should have a __str__ method),
    or None if not applicable.
    """

    @abstractmethod
    async def arecommend(
        self,
        problem: Problem,
        features_with_stats: Sequence[FeatureWithFullStats],
        dataset: Dataset,
        conn: DuckDBPyConnection,
    ) -> RecommendationsReport | None:
        """Generate recommendations for the given features.
        
        Args:
            problem: The problem definition including target information
            features_with_stats: Features with their computed statistics
            dataset: The dataset containing the data
            conn: Database connection for accessing secondary tables
            
        Returns:
            A structured report (RecommendationsReport), or None if this recommender
            cannot handle the given features/problem combination.
        """
        ...
