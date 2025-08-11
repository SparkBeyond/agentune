"""Tests for the numeric feature stats module."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
import pytest
from attrs import frozen

from agentune.analyze.context.base import ContextDefinition
from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.feature.base import FloatFeature
from agentune.analyze.feature.stats.stats import (
    NumericClassificationStats,
    NumericFeatureStats,
    NumericRegressionStats,
)
from agentune.analyze.feature.stats.stats_calculators import (
    NumericClassificationCalculator,
    NumericRegressionCalculator,
    NumericStatsCalculator,
)


# Simple Feature implementation for testing
@frozen
class SimpleNumericFeature(FloatFeature):
    """A simple Feature implementation for testing purposes."""

    name: str
    description: str = 'Test numeric feature'
    code: str = 'def evaluate(df): return df[self.name]'

    @property
    def params(self) -> Schema:
        # Schema constructor expects Field objects
        return Schema((Field(self.name, self.dtype),))

    @property
    def context_tables(self) -> Sequence[DuckdbTable]:
        return []

    @property
    def context_definitions(self) -> Sequence[ContextDefinition]:
        return []


class TestNumericFeatureStats:
    """Tests for numeric feature stats calculations."""

    def test_numeric_stats_with_valid_data(self) -> None:
        """Test that numeric stats are calculated correctly with valid data."""
        # Create test data
        series = pl.Series('test', [10.5, 20.3, 15.7, None, 0, 8.2])

        # Create a simple feature instance for testing
        feature = SimpleNumericFeature(name='test_feature')

        # Create calculator instance and calculate stats
        calculator = NumericStatsCalculator()
        stats = calculator.calculate_from_series(feature, series)

        # Verify results
        assert isinstance(stats, NumericFeatureStats)
        assert stats.n_total == 6
        assert stats.n_missing == 1
        assert stats.min == pytest.approx(0)
        assert stats.max == pytest.approx(20.3)
        assert stats.mean == pytest.approx(10.94)
        assert stats.median == pytest.approx(10.5)
        assert not np.isnan(stats.std)
        assert not np.isnan(stats.variance)
        assert stats.q1 < stats.median < stats.q3

    def test_numeric_stats_with_empty_data(self) -> None:
        """Test that numeric stats handle empty data correctly."""
        # Create empty series
        series = pl.Series('empty', [], dtype=pl.Float64)

        # Create a simple feature instance for testing
        feature = SimpleNumericFeature(name='test_feature')

        # Create calculator instance and calculate stats
        calculator = NumericStatsCalculator()
        stats = calculator.calculate_from_series(feature, series)

        # Verify results
        assert isinstance(stats, NumericFeatureStats)
        assert stats.n_total == 0
        assert stats.n_missing == 0
        assert np.isnan(stats.min)
        assert np.isnan(stats.max)
        assert np.isnan(stats.mean)
        assert np.isnan(stats.median)
        assert np.isnan(stats.std)
        assert np.isnan(stats.variance)
        assert np.isnan(stats.q1)
        assert np.isnan(stats.q3)

    def test_numeric_stats_with_all_nulls(self) -> None:
        """Test that numeric stats handle all null values correctly."""
        # Create series with all nulls
        series = pl.Series('all_null', [None, None, None])

        # Create a simple feature instance for testing
        feature = SimpleNumericFeature(name='test_feature')

        # Create calculator instance and calculate stats
        calculator = NumericStatsCalculator()
        stats = calculator.calculate_from_series(feature, series)

        # Verify results
        assert isinstance(stats, NumericFeatureStats)
        assert stats.n_total == 3
        assert stats.n_missing == 3
        assert np.isnan(stats.min)
        assert np.isnan(stats.max)
        assert np.isnan(stats.mean)
        assert np.isnan(stats.std)


class TestNumericRegressionStats:
    """Tests for numeric regression stats calculations."""

    def test_numeric_regression_stats(self) -> None:
        """Test that numeric regression stats are calculated correctly."""
        # Create test data - positively correlated
        feature = pl.Series('feature', [1.0, 2.0, 3.0, 4.0, None])
        target = pl.Series('target', [10.0, 20.0, 30.0, 40.0, 50.0])

        # Create a simple feature instance for testing
        feature_obj = SimpleNumericFeature(name='test_feature')

        # Create calculator instance and calculate stats
        calculator = NumericRegressionCalculator()
        stats = calculator.calculate_from_series(feature_obj, feature, target)

        # Verify results
        assert isinstance(stats, NumericRegressionStats)
        assert stats.n_samples == 4
        assert stats.n_missing_feature == 1
        assert stats.pearson_r == pytest.approx(1.0)  # Perfect positive correlation
        assert stats.pearson_p < 0.05  # Significant correlation
        assert stats.spearman_r == pytest.approx(1.0)  # Perfect rank correlation
        assert stats.spearman_p < 0.05  # Significant correlation

    def test_numeric_regression_stats_negative_correlation(self) -> None:
        """Test numeric regression stats with negative correlation."""
        # Create test data with negative correlation
        feature = pl.Series('feature', [4.0, 3.0, 2.0, 1.0])
        target = pl.Series('target', [10.0, 20.0, 30.0, 40.0])

        # Create a simple feature instance for testing
        feature_obj = SimpleNumericFeature(name='test_feature')

        # Create calculator instance and calculate stats
        calculator = NumericRegressionCalculator()
        stats = calculator.calculate_from_series(feature_obj, feature, target)

        # Verify results
        assert stats.pearson_r == pytest.approx(-1.0)  # Perfect negative correlation
        assert stats.spearman_r == pytest.approx(-1.0)  # Perfect negative rank correlation

    def test_numeric_regression_stats_no_correlation(self) -> None:
        """Test numeric regression stats with no correlation."""
        # Create test data with no correlation - constant target causes NaN correlation
        feature = pl.Series('feature', [1.0, 2.0, 3.0, 4.0])
        target = pl.Series('target', [25.0, 25.0, 25.0, 25.0])

        # Create a simple feature instance for testing
        feature_obj = SimpleNumericFeature(name='test_feature')

        # Create calculator instance and calculate stats
        calculator = NumericRegressionCalculator()
        stats = calculator.calculate_from_series(feature_obj, feature, target)

        # Verify results
        # With a constant target, pearson and spearman correlation will be NaN
        # scipy.stats.pearsonr returns NaN when one of the variables is constant
        assert np.isnan(stats.pearson_r)  # NaN correlation with constant target
        assert np.isnan(stats.spearman_r)  # NaN rank correlation with constant target


class TestNumericClassificationStats:
    """Tests for numeric classification stats calculations."""

    def test_numeric_classification_stats(self) -> None:
        """Test that numeric classification stats are calculated correctly."""
        # Create test data - feature values differ by class
        feature = pl.Series('feature', [1.0, 1.2, 5.0, 5.2, None, 10.0, 10.2])
        target = pl.Series('target', ['A', 'A', 'B', 'B', 'C', 'C', 'C'])

        # Create a simple feature instance for testing
        feature_obj = SimpleNumericFeature(name='test_feature')

        # Create calculator instance and calculate stats
        calculator = NumericClassificationCalculator()
        stats = calculator.calculate_from_series(feature_obj, feature, target)

        # Verify results
        assert isinstance(stats, NumericClassificationStats)
        assert stats.n_samples == 6
        assert stats.n_missing_feature == 1
        assert not np.isnan(stats.anova_f)  # Should have significant F statistic
        assert not np.isnan(stats.p_value)  # Should have valid p-value

    def test_numeric_classification_stats_no_separation(self) -> None:
        """Test numeric classification stats when feature doesn't separate classes."""
        # Create test data with no class separation
        feature = pl.Series('feature', [5.0, 5.0, 5.0, 5.0])
        target = pl.Series('target', ['A', 'B', 'C', 'D'])

        # Create a simple feature instance for testing
        feature_obj = SimpleNumericFeature(name='test_feature')

        # Create calculator instance and calculate stats
        calculator = NumericClassificationCalculator()
        stats = calculator.calculate_from_series(feature_obj, feature, target)

        # Verify results
        assert np.isnan(stats.anova_f)  # F-stat is nan for identical values
        assert np.isnan(stats.p_value)  # p-value is nan
