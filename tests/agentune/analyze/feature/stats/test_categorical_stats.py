"""Tests for the categorical feature stats module."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
import pytest

from agentune.analyze.context.base import ContextDefinition
from agentune.analyze.core.database import DatabaseTable
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.feature.base import CategoricalFeature
from agentune.analyze.feature.stats.stats import (
    CategoricalClassificationStats,
    CategoricalFeatureStats,
    CategoricalRegressionStats,
)
from agentune.analyze.feature.stats.stats_calculators import (
    CategoricalClassificationCalculator,
    CategoricalRegressionCalculator,
    CategoricalStatsCalculator,
)


# Simple Feature implementation for testing
class SimpleCategoricalFeature(CategoricalFeature):
    """A simple Feature implementation for testing purposes."""

    def __init__(self, name: str, categories: tuple[str, ...]):
        self._name = name
        self._categories = categories
        self._description = 'Test categorical feature'
        self._code = 'def evaluate(df): return df[self.name]'

    @property
    def categories(self) -> Sequence[str]:
        return self._categories

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def code(self) -> str:
        return self._code

    @property
    def params(self) -> Schema:
        # Schema constructor expects Field objects
        return Schema((Field(self._name, self.dtype),))

    @property
    def context_tables(self) -> Sequence[DatabaseTable]:
        return []

    @property
    def context_objects(self) -> Sequence[ContextDefinition]:
        return []


class TestCategoricalFeatureStats:
    """Tests for categorical feature stats calculations."""

    def test_categorical_stats_with_valid_data(self) -> None:
        """Test that categorical stats are calculated correctly with valid data."""
        # Create test data
        series = pl.Series('test', ['A', 'B', 'A', None, 'C', 'A'])

        # Create a simple feature instance for testing
        feature = SimpleCategoricalFeature(name='test_feature', categories=('A', 'B', 'C'))

        # Create calculator instance and calculate stats
        calculator = CategoricalStatsCalculator()
        stats = calculator.calculate_from_series(feature, series)

        # Verify results
        assert isinstance(stats, CategoricalFeatureStats)
        assert stats.n_total == 6
        assert stats.n_missing == 1
        assert stats.unique_count == 3  # A, B, C
        assert stats.value_counts.get('A', 0) == 3
        assert stats.value_counts.get('B', 0) == 1
        assert stats.value_counts.get('C', 0) == 1

    def test_categorical_stats_with_empty_data(self) -> None:
        """Test that categorical stats handle empty data correctly."""
        # Create empty series
        series = pl.Series('empty', [], dtype=pl.Utf8)

        # Create a simple feature instance for testing
        feature = SimpleCategoricalFeature(name='test_feature', categories=('A', 'B', 'C'))

        # Create calculator instance and calculate stats
        calculator = CategoricalStatsCalculator()
        stats = calculator.calculate_from_series(feature, series)

        # Verify results
        assert isinstance(stats, CategoricalFeatureStats)
        assert stats.n_total == 0
        assert stats.n_missing == 0
        assert stats.unique_count == 0
        assert len(stats.value_counts) == 3 # Still has all categories, with 0 counts

    def test_categorical_stats_with_all_nulls(self) -> None:
        """Test that categorical stats handle all null values correctly."""
        # Create series with all nulls
        series = pl.Series('all_null', [None, None, None])

        # Create a simple feature instance for testing
        feature = SimpleCategoricalFeature(name='test_feature', categories=('A', 'B', 'C'))

        # Create calculator instance and calculate stats
        calculator = CategoricalStatsCalculator()
        stats = calculator.calculate_from_series(feature, series)

        # Verify results
        assert isinstance(stats, CategoricalFeatureStats)
        assert stats.n_total == 3
        assert stats.n_missing == 3

        # Should have zero unique non-null values
        assert stats.unique_count == 0
        assert stats.value_counts == {'A': 0, 'B': 0, 'C': 0}


class TestCategoricalRegressionStats:
    """Tests for categorical regression stats calculations."""

    def test_categorical_regression_stats(self) -> None:
        """Test that categorical regression stats are calculated correctly."""
        # Create test data
        feature = pl.Series('feature', ['A', 'B', 'A', None, 'C', 'A'])
        target = pl.Series('target', [10.0, 5.0, 12.0, 8.0, 15.0, 9.0])

        # Create a simple feature instance for testing
        feature_obj = SimpleCategoricalFeature(name='test_feature', categories=('A', 'B', 'C'))

        # Create calculator instance and calculate stats
        calculator = CategoricalRegressionCalculator()
        stats = calculator.calculate_from_series(feature_obj, feature, target)

        # Verify results
        assert isinstance(stats, CategoricalRegressionStats)
        assert stats.n_samples == 5
        assert stats.n_missing_feature == 1
        assert len(stats.categories) == 3  # A, B, C
        assert len(stats.category_counts) == 3
        assert len(stats.mean_by_category) == 3
        assert len(stats.shift_by_category) == 3

        # Check specific category stats
        a_index = list(stats.categories).index('A') if 'A' in stats.categories else -1
        if a_index != -1:
            assert stats.category_counts[a_index] == 3
            assert stats.mean_by_category[a_index] == pytest.approx((10.0 + 12.0 + 9.0) / 3)

    def test_categorical_regression_stats_with_single_category(self) -> None:
        """Test categorical regression stats with a single category."""
        # Create test data with single category
        feature = pl.Series('feature', ['A', 'A', 'A', 'A'])
        target = pl.Series('target', [10.0, 12.0, 9.0, 11.0])

        # Create a simple feature instance for testing
        feature_obj = SimpleCategoricalFeature(name='test_feature', categories=('A',))

        # Create calculator instance and calculate stats
        calculator = CategoricalRegressionCalculator()
        stats = calculator.calculate_from_series(feature_obj, feature, target)

        # Verify results
        assert stats.n_samples == 4
        assert len(stats.categories) == 1
        assert stats.categories[0] == 'A'
        assert stats.category_counts[0] == 4
        assert stats.mean_by_category[0] == pytest.approx(10.5)
        assert stats.shift_by_category[0] == pytest.approx(
            0.0
        )  # No shift since all values belong to same category


class TestCategoricalClassificationStats:
    """Tests for categorical classification stats calculations."""

    def test_categorical_classification_stats(self) -> None:
        """Test that categorical classification stats are calculated correctly."""
        # Create test data - contingency table-like relationship using only numeric data
        feature = pl.Series('feature', ['0', '0', '1', '1', '2', '2'])  # Feature categories
        target = pl.Series('target', [0, 0, 1, 1, 2, 2])  # Numeric target categories

        # Create a simple feature instance for testing
        feature_obj = SimpleCategoricalFeature(name='test_feature', categories=('0', '1', '2'))

        # Create calculator instance and calculate stats
        calculator = CategoricalClassificationCalculator()
        stats = calculator.calculate_from_series(feature_obj, feature, target)

        # Verify results
        assert isinstance(stats, CategoricalClassificationStats)
        assert stats.n_samples == 6
        assert stats.n_missing_feature == 0  # No missing values
        assert len(stats.categories) == 3  # 0, 1, 2
        assert len(stats.classes) == 3  # 0, 1, 2
        assert len(stats.category_class_counts) == 3  # One row per category
        assert len(stats.category_class_counts[0]) == 3  # One column per class
        assert not np.isnan(stats.info_gain)  # Should have non-NaN info gain
        assert len(stats.lift) == 3  # One row per category
        assert len(stats.lift[0]) == 3  # One column per class

        # Check specific counts in the contingency matrix
        x_index = list(stats.categories).index('0') if '0' in stats.categories else -1
        a_index = list(stats.classes).index('0') if '0' in stats.classes else -1
        if x_index != -1 and a_index != -1:
            assert (
                stats.category_class_counts[x_index][a_index] == 2
            )  # 2 instances of feature category 0 with target 0

    def test_categorical_classification_stats_with_perfect_correlation(self) -> None:
        """Test categorical classification with perfect correlation between feature and target."""
        # Create test data with perfect relationship using numeric data
        feature = pl.Series('feature', ['0', '1', '2'])
        target = pl.Series('target', ['0', '1', '2'])

        # Create a simple feature instance for testing
        feature_obj = SimpleCategoricalFeature(name='test_feature', categories=('0', '1', '2'))

        # Create calculator instance and calculate stats
        calculator = CategoricalClassificationCalculator()
        stats = calculator.calculate_from_series(feature_obj, feature, target)

        # Verify results
        assert stats.n_samples == 3
        assert len(stats.categories) == 3
        assert len(stats.classes) == 3

        # Perfect relationship should have high info gain
        assert not np.isnan(stats.info_gain)

        # Check contingency matrix - should be diagonal
        x_index = list(stats.categories).index('0')
        a_index = list(stats.classes).index('0')
        y_index = list(stats.categories).index('1')
        b_index = list(stats.classes).index('1')
        z_index = list(stats.categories).index('2')
        c_index = list(stats.classes).index('2')

        assert stats.category_class_counts[x_index][a_index] == 1
        assert stats.category_class_counts[y_index][b_index] == 1
        assert stats.category_class_counts[z_index][c_index] == 1
