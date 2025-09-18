"""Tests for regression statistics functionality - focused on regression-specific features.

Covers:
- Histogram format validation (numpy-style format)
- Numeric regression feature statistics (histogram creation)
- Correlation calculations (Pearson and Spearman)
- Regression stats usage conditions
"""
from __future__ import annotations

from typing import Any

import polars as pl
from attrs import frozen

from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.core.types import float64, int32
from agentune.analyze.feature.base import SyncFloatFeature
from agentune.analyze.feature.problem import (
    ClassificationProblem,
    ProblemDescription,
    RegressionProblem,
)
from agentune.analyze.feature.stats.stats_calculators import (
    NumericRegressionRelationshipStatsCalculator,
    NumericRegressionStatsCalculator,
    should_use_regression_stats,
)


# Helper functions for creating test objects
def _regression_problem() -> RegressionProblem:
    """Create a RegressionProblem for testing."""
    problem_desc = ProblemDescription(target_column='target')
    return RegressionProblem(
        problem_description=problem_desc,
        target_column=Field('target', float64)
    )


def _classification_problem() -> ClassificationProblem:
    """Create a ClassificationProblem for testing."""
    problem_desc = ProblemDescription(target_column='target')
    return ClassificationProblem(
        problem_description=problem_desc,
        target_column=Field('target', int32),
        classes=(0, 1)
    )


# Simple Feature implementation for testing
@frozen
class MockFloatFeature(SyncFloatFeature):
    name: str = 'test_feature'
    description: str = 'Test feature'
    technical_description: str = 'Test technical description'
    default_for_missing: float = 0.0
    default_for_nan: float = 0.0
    default_for_infinity: float = 0.0
    default_for_neg_infinity: float = 0.0
    
    @property
    def params(self) -> Schema:
        return Schema((Field('test_col', float64),))
    
    @property
    def secondary_tables(self) -> list[Any]:
        return []
    
    @property
    def join_strategies(self) -> list[Any]:
        return []
    
    def evaluate(self, args: Any, conn: Any) -> Any:  # noqa: ARG002
        return args[0]



# ------------------------------
# A. Histogram format validation
# ------------------------------

def test_histogram_format_structure() -> None:
    """Test that histogram follows numpy.histogram format."""
    calculator = NumericRegressionStatsCalculator(n_histogram_bins=3)
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    series = pl.Series('test', values)
    float_feature = MockFloatFeature()
    
    stats = calculator.calculate_from_series(float_feature, series)
    
    # Test numpy histogram format: counts and bin_edges
    assert isinstance(stats.histogram_counts, tuple)
    assert isinstance(stats.histogram_bin_edges, tuple)
    assert len(stats.histogram_bin_edges) == len(stats.histogram_counts) + 1
    assert all(isinstance(c, int) for c in stats.histogram_counts)
    assert all(isinstance(e, float) for e in stats.histogram_bin_edges)


def test_histogram_creation_with_various_bins() -> None:
    """Test histogram creation for numeric features with different bin counts."""
    float_feature = MockFloatFeature()
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    series = pl.Series('test', values)
    
    # Test with 4 bins
    calculator = NumericRegressionStatsCalculator(n_histogram_bins=4)
    stats = calculator.calculate_from_series(float_feature, series)
    
    # Test standard numpy histogram format
    assert len(stats.histogram_counts) == 4
    assert len(stats.histogram_bin_edges) == 5  # n_bins + 1
    assert sum(stats.histogram_counts) == len(values)
    
    # Verify bin edges are monotonically increasing
    edges = list(stats.histogram_bin_edges)
    assert all(edges[i] <= edges[i+1] for i in range(len(edges)-1))


def test_histogram_empty_data() -> None:
    """Test histogram behavior with empty or non-numeric data."""
    float_feature = MockFloatFeature()
    calculator = NumericRegressionStatsCalculator(n_histogram_bins=4)
    
    # Empty series
    empty_series = pl.Series('test', [], dtype=pl.Float64)
    stats = calculator.calculate_from_series(float_feature, empty_series)
    assert stats.histogram_counts == ()
    assert stats.histogram_bin_edges == ()
    
    # Series with only nulls
    null_series = pl.Series('test', [None, None, None])
    stats = calculator.calculate_from_series(float_feature, null_series)
    assert stats.histogram_counts == ()
    assert stats.histogram_bin_edges == ()


# -----------------------------------
# B. Correlation calculations
# -----------------------------------

def test_perfect_positive_correlation() -> None:
    """Test perfect positive correlation calculation."""
    float_feature = MockFloatFeature()
    regression_problem = _regression_problem()
    calculator = NumericRegressionRelationshipStatsCalculator()
    
    # Perfect positive correlation
    feature_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    target_values = [3.0, 5.0, 7.0, 9.0, 11.0]
    feature_series = pl.Series('feature', feature_values)
    target_series = pl.Series('target', target_values)
    
    stats = calculator.calculate_from_series(
        float_feature, feature_series, target_series, regression_problem
    )
    
    assert abs(stats.pearson_correlation - 1.0) < 0.01
    assert abs(stats.spearman_correlation - 1.0) < 0.01


def test_perfect_negative_correlation() -> None:
    """Test perfect negative correlation calculation."""
    float_feature = MockFloatFeature()
    regression_problem = _regression_problem()
    calculator = NumericRegressionRelationshipStatsCalculator()
    
    # Perfect negative correlation
    feature_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    target_values = [11.0, 9.0, 7.0, 5.0, 3.0]
    feature_series = pl.Series('feature', feature_values)
    target_series = pl.Series('target', target_values)
    
    stats = calculator.calculate_from_series(
        float_feature, feature_series, target_series, regression_problem
    )
    
    assert abs(stats.pearson_correlation + 1.0) < 0.01
    assert abs(stats.spearman_correlation + 1.0) < 0.01


def test_zero_correlation() -> None:
    """Test zero correlation with independent data."""
    float_feature = MockFloatFeature()
    regression_problem = _regression_problem()
    calculator = NumericRegressionRelationshipStatsCalculator()
    
    # Independent feature and target (no correlation)
    feature_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    target_values = [5.0, 2.0, 8.0, 1.0, 7.0, 3.0, 6.0, 4.0] 
    feature_series = pl.Series('feature', feature_values)
    target_series = pl.Series('target', target_values)
    
    stats = calculator.calculate_from_series(
        float_feature, feature_series, target_series, regression_problem
    )
    
    # Should be close to zero (not exactly due to small sample)
    assert abs(stats.pearson_correlation) < 0.3
    assert abs(stats.spearman_correlation) < 0.3


def test_correlation_with_insufficient_data() -> None:
    """Test correlation calculation with insufficient data points."""
    float_feature = MockFloatFeature()
    regression_problem = _regression_problem()
    calculator = NumericRegressionRelationshipStatsCalculator()
    
    # Only one data point
    feature_series = pl.Series('feature', [1.0])
    target_series = pl.Series('target', [2.0])
    
    # Should raise ValueError when insufficient data (<=1 samples)
    try:
        calculator.calculate_from_series(
            float_feature, feature_series, target_series, regression_problem
        )
        raise AssertionError('Expected ValueError to be raised for insufficient data')
    except ValueError as e:
        assert 'Insufficient samples' in str(e)  # noqa: PT017


# -----------------------------------
# C. Regression stats usage conditions
# -----------------------------------

def test_should_use_regression_stats_positive_case() -> None:
    """Test when regression stats should be used (positive case)."""
    float_feature = MockFloatFeature()
    regression_problem = _regression_problem()
    
    assert should_use_regression_stats(float_feature, regression_problem) is True


def test_should_use_regression_stats_negative_cases() -> None:
    """Test when regression stats should NOT be used."""
    float_feature = MockFloatFeature()
    
    # Non-regression problem should return False
    classification_problem = _classification_problem()
    assert should_use_regression_stats(float_feature, classification_problem) is False
