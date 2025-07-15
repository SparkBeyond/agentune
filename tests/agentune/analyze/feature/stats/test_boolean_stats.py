"""Tests for the boolean feature stats module."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import polars as pl
import pytest

from agentune.analyze.context.base import ContextDefinition
from agentune.analyze.core.database import DatabaseTable
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.feature.base import BoolFeature
from agentune.analyze.feature.stats.stats import (
    BooleanClassificationStats,
    BooleanFeatureStats,
    BooleanRegressionStats,
)
from agentune.analyze.feature.stats.stats_calculators import (
    BooleanClassificationCalculator,
    BooleanRegressionCalculator,
    BooleanStatsCalculator,
)


# Simple Feature implementation for testing
class SimpleBoolFeature(BoolFeature):
    """A simple Feature implementation for testing purposes."""

    def __init__(self, name: str):
        self._name = name
        self._description = 'Test bool feature'
        self._code = 'def evaluate(df): return df[self.name]'

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
    def context_tables(self) -> Iterable[DatabaseTable]:
        return []

    @property
    def context_objects(self) -> Iterable[ContextDefinition]:
        return []


class TestBooleanFeatureStats:
    """Tests for boolean feature stats calculations."""

    def test_boolean_stats_with_valid_data(self) -> None:
        """Test that boolean stats are calculated correctly with valid data."""
        # Create test data
        series = pl.Series('test', [True, False, True, None, True])

        # Create a simple feature instance for testing
        feature = SimpleBoolFeature(name='test_feature')

        # Create calculator instance and calculate stats
        calculator = BooleanStatsCalculator()
        stats = calculator.calculate_from_series(feature, series)

        # Verify results
        assert isinstance(stats, BooleanFeatureStats)
        assert stats.n_total == 5
        assert stats.n_missing == 1
        assert stats.support == 3
        assert stats.coverage == 0.75  # 3/4

    def test_boolean_stats_with_empty_data(self) -> None:
        """Test that boolean stats handle empty data correctly."""
        # Create empty series
        series = pl.Series('empty', [], dtype=pl.Boolean)

        # Create a simple feature instance for testing
        feature = SimpleBoolFeature(name='test_feature')

        # Create calculator instance and calculate stats
        calculator = BooleanStatsCalculator()
        stats = calculator.calculate_from_series(feature, series)

        # Verify results
        assert isinstance(stats, BooleanFeatureStats)
        assert stats.n_total == 0
        assert stats.n_missing == 0
        assert stats.support == 0
        assert np.isnan(stats.coverage)

    def test_boolean_stats_with_all_nulls(self) -> None:
        """Test that boolean stats handle all null values correctly."""
        # Create series with all nulls
        series = pl.Series('all_null', [None, None, None])

        # Create a simple feature instance for testing
        feature = SimpleBoolFeature(name='test_feature')

        # Create calculator instance and calculate stats
        calculator = BooleanStatsCalculator()
        stats = calculator.calculate_from_series(feature, series)

        # Verify results
        assert isinstance(stats, BooleanFeatureStats)
        assert stats.n_total == 3
        assert stats.n_missing == 3
        assert stats.support == 0
        assert np.isnan(stats.coverage)


class TestBooleanRegressionStats:
    """Tests for boolean regression stats calculations."""

    def test_boolean_regression_stats(self) -> None:
        """Test that boolean regression stats are calculated correctly."""
        # Create test data
        feature = pl.Series('feature', [True, False, True, None, True])
        target = pl.Series('target', [10.0, 5.0, 12.0, 8.0, 9.0])

        # Create a simple feature instance for testing
        feature_obj = SimpleBoolFeature(name='test_feature')

        # Create calculator instance and calculate stats
        calculator = BooleanRegressionCalculator()
        stats = calculator.calculate_from_series(feature_obj, feature, target)

        # Verify results
        assert isinstance(stats, BooleanRegressionStats)
        assert stats.n_samples == 4
        assert stats.n_missing_feature == 1
        assert stats.mean_true == pytest.approx(10.33333, abs=1e-5)
        assert stats.mean_false == 5.0
        assert stats.shift == pytest.approx(1.33333, abs=1e-5)  # mean_true - overall_mean
        assert not np.isnan(stats.point_biserial)

    def test_boolean_regression_stats_with_all_true(self) -> None:
        """Test boolean regression stats with all True values."""
        # Create test data with all True
        feature = pl.Series('feature', [True, True, True])
        target = pl.Series('target', [10.0, 12.0, 9.0])

        # Create a simple feature instance for testing
        feature_obj = SimpleBoolFeature(name='test_feature')

        # Create calculator instance and calculate stats
        calculator = BooleanRegressionCalculator()
        stats = calculator.calculate_from_series(feature_obj, feature, target)

        # Verify results
        assert stats.n_samples == 3
        assert stats.mean_true == pytest.approx(10.33333, abs=1e-5)
        # When all values are True, the following measures are NaN
        assert np.isnan(stats.mean_false), 'mean_false should be NaN with all True values'
        assert stats.shift == 0, 'shift should be 0 when all values are one class'
        assert np.isnan(stats.point_biserial), 'point_biserial should be NaN for constant features'


class TestBooleanClassificationStats:
    """Tests for boolean classification stats calculations."""

    def test_boolean_classification_stats(self) -> None:
        """Test that boolean classification stats are calculated correctly."""
        # Create test data
        feature = pl.Series('feature', [True, False, True, None, True])
        target = pl.Series('target', ['A', 'B', 'A', 'B', 'C'])

        # Create a simple feature instance for testing
        feature_obj = SimpleBoolFeature(name='test_feature')

        # Create calculator instance and calculate stats
        calculator = BooleanClassificationCalculator()
        stats = calculator.calculate_from_series(feature_obj, feature, target)

        # Verify results
        assert isinstance(stats, BooleanClassificationStats)
        assert stats.n_samples == 4
        assert stats.n_missing_feature == 1
        assert len(stats.categories) == 3  # A, B, C
        assert len(stats.true_counts) == 3  # Counts for True values in each category
        assert not np.isnan(stats.info_gain)  # Exact value depends on sklearn implementation
        assert len(stats.lift) == 3  # Lift values for each category
        assert len(stats.shift) == 3  # Shift values for each category

    def test_boolean_classification_stats_with_binary_target(self) -> None:
        """Test boolean classification stats with binary target."""
        # Create test data with binary target
        feature = pl.Series('feature', [True, False, True, False])
        target = pl.Series('target', [1, 0, 1, 0])

        # Create a simple feature instance for testing
        feature_obj = SimpleBoolFeature(name='test_feature')

        # Create calculator instance and calculate stats
        calculator = BooleanClassificationCalculator()
        stats = calculator.calculate_from_series(feature_obj, feature, target)

        # Verify results
        assert stats.n_samples == 4
        assert len(stats.categories) == 2  # 0, 1
        assert len(stats.true_counts) == 2  # Counts for True values in each category
        assert stats.true_counts[1] == 2  # 2 True values in category 1
        assert stats.true_counts[0] == 0  # 0 True values in category 0
        assert not np.isnan(stats.info_gain)  # Should be informative
