import asyncio
from abc import ABC, abstractmethod
from typing import override

import polars as pl
from attrs import frozen

from agentune.analyze.core.dataset import Dataset
from agentune.analyze.feature.base import Feature
from agentune.analyze.feature.problem import Problem


# --- Generic base classes ---
class FeatureStats[F: Feature]:
    """Base class for all feature statistics."""

    n_total: int  # Total number of data points
    n_missing: int  # Number of missing values in the feature


class RelationshipStats[F: Feature]:
    """Base class for statistics that describe the relationship between a feature and a target."""

    n_samples: int  # Number of samples used in the calculation
    n_missing_feature: int  # Number of samples where the feature was missing


class FeatureStatsCalculator[F: Feature](ABC):
    @abstractmethod
    async def acalculate_from_series(self, feature: F, series: pl.Series) -> FeatureStats[F]:
        """Calculate feature statistics from a single polars Series asynchronously.

        Args:
            feature: The feature to calculate statistics for
            series: A polars Series containing the feature data

        Returns:
            Feature statistics object

        """
        ...

    async def acalculate_from_dataset(
        self, feature: F, dataset: Dataset, feature_col: str
    ) -> FeatureStats[F]:
        """Simple implementation for calculating feature statistics from a dataset asynchronously.

        This implementation simply calls `calculate_from_series` with the relevant column of the dataset.

        Args:
            feature: The feature to calculate statistics for
            dataset: The dataset containing the feature
            feature_col: The name of the feature column

        Returns:
            Feature statistics object

        """
        series = dataset.data[feature_col]
        return await self.acalculate_from_series(feature, series)

class SyncFeatureStatsCalculator[F: Feature](FeatureStatsCalculator[F]):
    """Synchronous calculator for feature statistics only."""

    @abstractmethod
    def calculate_from_series(self, feature: F, series: pl.Series) -> FeatureStats[F]:
        """Calculate feature statistics from a single polars Series.

        Args:
            feature: The feature to calculate statistics for
            series: A polars Series containing the feature data

        Returns:
            Feature statistics object

        """

    @override
    async def acalculate_from_series(self, feature: F, series: pl.Series) -> FeatureStats[F]:
        return await asyncio.to_thread(self.calculate_from_series, feature, series.clone())

    def calculate_from_dataset(
        self, feature: F, dataset: Dataset, feature_col: str
    ) -> FeatureStats[F]:
        """Simple implementation for calculating feature statistics from a dataset.

        This implementation simply calls `calculate_from_series` with the relevant column of the dataset.

        Args:
            feature: The feature to calculate statistics for
            dataset: The dataset containing the feature
            feature_col: The name of the feature column

        Returns:
            Feature statistics object

        """
        series = dataset.data[feature_col]
        return self.calculate_from_series(feature, series)
    
    @override
    async def acalculate_from_dataset(
        self, feature: F, dataset: Dataset, feature_col: str
    ) -> FeatureStats[F]:
        return await asyncio.to_thread(self.calculate_from_dataset, feature, dataset.copy_to_thread(), feature_col)


# Relationship Stats Calculators (feature-target interactions)
class RelationshipStatsCalculator[F: Feature](ABC):
    """Calculator for computing feature-target relationship statistics.

    Type parameters:
    F: The type of features to calculate statistics for
    T: The type of target (classification or regression)
    """
    
    @abstractmethod
    async def acalculate_from_series(
        self, feature: F, series: pl.Series, target: pl.Series, problem: Problem
    ) -> RelationshipStats[F]:
        """Calculate relationship statistics from feature and target polars Series asynchronously.

        Args:
            feature: The feature to calculate statistics for
            series: A polars Series containing the feature data
            target: A polars Series containing the target data
            problem: problem definition including the target column name and list of classes

        Returns:
            Relationship statistics object

        """
        ...

    @abstractmethod
    async def acalculate_from_dataset(
        self, feature: F, dataset: Dataset, feature_col: str, problem: Problem
    ) -> RelationshipStats[F]:
        """Calculate relationship statistics from a dataset stream asynchronously.

        Args:
            feature: The feature to calculate statistics for
            dataset: The dataset stream containing the feature and target
            feature_col: The name of the feature column
            problem: problem definition including the target column name and list of classes

        Returns:
            Relationship statistics object

        """
        ...


class SyncRelationshipStatsCalculator[F: Feature](RelationshipStatsCalculator[F]):
    """Synchronous calculator for feature-target relationship statistics."""

    @abstractmethod
    def calculate_from_series(
        self, feature: F, series: pl.Series, target: pl.Series, problem: Problem
    ) -> RelationshipStats[F]:
        """Calculate relationship statistics from feature and target polars Series.

        Args:
            feature: The feature to calculate statistics for
            series: A polars Series containing the feature data
            target: A polars Series containing the target data
            problem: problem definition including the target column name and list of classes

        Returns:
            Relationship statistics object

        """
        ...

    @override
    async def acalculate_from_series(self, feature: F, series: pl.Series, target: pl.Series,
                                     problem: Problem) -> RelationshipStats[F]:
        return await asyncio.to_thread(self.calculate_from_series, feature, series.clone(), target.clone(), problem)


    def calculate_from_dataset(
        self, feature: F, dataset: Dataset, feature_col: str, problem: Problem
    ) -> RelationshipStats[F]:
        """Simple implementation for calculating relationship statistics from a dataset.

        This implementation simply calls `calculate_from_series` with the relevant columns of the dataset.

        Args:
            feature: The feature to calculate statistics for
            dataset: The dataset containing the feature and target
            feature_col: The name of the feature column
            problem: problem definition including the target column name and list of classes

        Returns:
            Relationship statistics object

        """
        feature_series = dataset.data[feature_col]
        target_series = dataset.data[problem.target_column.name]
        return self.calculate_from_series(feature, feature_series, target_series, problem)

    @override
    async def acalculate_from_dataset(
        self, feature: F, dataset: Dataset, feature_col: str, problem: Problem
    ) -> RelationshipStats[F]:
        return await asyncio.to_thread(self.calculate_from_dataset, feature, dataset.copy_to_thread(), feature_col, problem)


# --- Bundle all together generically ---

@frozen
class FullFeatureStats[F: Feature]:
    """Container that combines both feature statistics and relationship statistics."""

    feature: FeatureStats[F]  # Statistics about the feature itself
    relationship: RelationshipStats[F]  # Statistics about feature-target relationship


@frozen
class FeatureWithFullStats[F: Feature]:
    feature: F
    stats: FullFeatureStats[F]
