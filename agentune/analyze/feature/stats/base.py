import asyncio
from abc import ABC, abstractmethod
from typing import override

import polars as pl
from attrs import field, frozen
from frozendict import frozendict

from agentune.analyze.core.dataset import Dataset
from agentune.analyze.feature.base import Feature
from agentune.analyze.feature.problem import Problem
from agentune.analyze.util.attrutil import frozendict_converter
from agentune.analyze.util.cattrutil import UseTypeTag

CategoryClassMatrix = tuple[tuple[float, ...], ...]


@frozen
class BinningInfo:
    """Information about how numeric data was binned into categories."""
    bin_edges: tuple[float, ...]  # Bin boundaries

    def n_bins(self) -> int:
        return len(self.bin_edges) - 1


# --- Generic base classes ---
@frozen
class FeatureStats(UseTypeTag):
    """Base class for all feature statistics."""

    n_total: int  # total number of data points (including missing)
    n_missing: int  # number of missing values
    categories: tuple[str, ...]  # Categories, either native or binned numeric
    value_counts: frozendict[str, int] = field(converter=frozendict_converter)  # Value counts per category
    # Per-category support as percentages (sum to 1.0 over non-missing rows), essentially normalized value counts
    support: tuple[float, ...]


@frozen
class RelationshipStats(UseTypeTag):
    """Base class for statistics that describe the relationship between a feature and a target.

    For numeric features, values are automatically binned into quantile-based categories
    (default: 5 bins labeled as Q1, Q2, Q3, Q4, Q5) before calculating relationship statistics.
    This means lift and mean_shift matrices represent relationships between these binned
    categories and target classes, not the raw numeric values.

    For categorical features, the original categories are used directly.
    """
    n_total: int
    n_missing: int
    # Target-related
    classes: tuple[str, ...]  # The target classes
    # Lift matrix: shape (num_categories, num_classes)
    # Definition: lift[i,j] = P(class_j | category_i) / P(class_j)
    lift: CategoryClassMatrix
    # P(class|category) - P(class) → (k, c)
    # Definition: mean_shift[i,j] = P(class_j | category_i) - P(class_j)
    mean_shift: CategoryClassMatrix

    # SSE reduction when using this feature to predict target via linear regression
    # Definition: baseline_stdev - feature_stdev (higher values indicate better predictive power)
    sse_reduction: float

    # R² (coefficient of determination) - proportion of variance explained by the feature
    # Definition: 1 - (SSE_with_feature / SSE_baseline)
    # Range: [0, 1] where 0 = no predictive power, 1 = perfect prediction
    r_squared: float

    # Binning information for numeric features (None for categorical features)
    binning_info: BinningInfo | None

    # Amount of data points per target class (integers)
    # Definition: for each class c, (# rows where target==c)
    totals_per_class: tuple[int, ...]

    # per-class missing percentages (floats)
    # Definition: for each class c, (# missing where target==c) / (# rows where target==c)
    missing_percentage_per_class: tuple[float, ...]


class FeatureStatsCalculator(ABC):
    @abstractmethod
    async def acalculate_from_series(self, feature: Feature, series: pl.Series) -> FeatureStats:
        """Calculate feature statistics from a single polars Series asynchronously.

        Args:
            feature: The feature to calculate statistics for
            series: A polars Series containing the feature data

        Returns:
            Feature statistics object

        """
        ...

    async def acalculate_from_dataset(
        self, feature: Feature, dataset: Dataset, feature_col: str
    ) -> FeatureStats:
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


class SyncFeatureStatsCalculator(FeatureStatsCalculator):
    """Synchronous calculator for feature statistics only."""

    @abstractmethod
    def calculate_from_series(self, feature: Feature, series: pl.Series) -> FeatureStats:
        """Calculate feature statistics from a single polars Series.

        Args:
            feature: The feature to calculate statistics for
            series: A polars Series containing the feature data

        Returns:
            Feature statistics object

        """

    @override
    async def acalculate_from_series(self, feature: Feature, series: pl.Series) -> FeatureStats:
        return await asyncio.to_thread(self.calculate_from_series, feature, series.clone())

    def calculate_from_dataset(
        self, feature: Feature, dataset: Dataset, feature_col: str
    ) -> FeatureStats:
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
        self, feature: Feature, dataset: Dataset, feature_col: str
    ) -> FeatureStats:
        return await asyncio.to_thread(self.calculate_from_dataset, feature, dataset.copy_to_thread(), feature_col)


# Relationship Stats Calculators (feature-target interactions)
class RelationshipStatsCalculator(ABC):
    """Calculator for computing feature-target relationship statistics."""
    
    @abstractmethod
    async def acalculate_from_series(
        self, feature: Feature, series: pl.Series, target: pl.Series, problem: Problem
    ) -> RelationshipStats:
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
        self, feature: Feature, dataset: Dataset, feature_col: str, problem: Problem
    ) -> RelationshipStats:
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


class SyncRelationshipStatsCalculator(RelationshipStatsCalculator):
    """Synchronous calculator for feature-target relationship statistics."""

    @abstractmethod
    def calculate_from_series(
        self, feature: Feature, series: pl.Series, target: pl.Series, problem: Problem
    ) -> RelationshipStats:
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
    async def acalculate_from_series(self, feature: Feature, series: pl.Series, target: pl.Series, problem: Problem) -> RelationshipStats:
        return await asyncio.to_thread(self.calculate_from_series, feature, series.clone(), target.clone(), problem)


    def calculate_from_dataset(
        self, feature: Feature, dataset: Dataset, feature_col: str, problem: Problem
    ) -> RelationshipStats:
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
        self, feature: Feature, dataset: Dataset, feature_col: str, problem: Problem
    ) -> RelationshipStats:
        return await asyncio.to_thread(self.calculate_from_dataset, feature, dataset.copy_to_thread(), feature_col, problem)


# --- Bundle all together generically ---

@frozen
class FullFeatureStats:
    """Container that combines both feature statistics and relationship statistics."""

    feature: FeatureStats  # Statistics about the feature itself
    relationship: RelationshipStats  # Statistics about feature-target relationship


@frozen
class FeatureWithFullStats:
    feature: Feature
    stats: FullFeatureStats
