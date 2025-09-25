"""Relationship stats tests for UnifiedRelationshipStatsCalculator.

Covers:
- Type coverage across feature types (bool, numeric, categorical) × target types (categorical, numeric)
- Binning behavior for numeric features/targets (labels, shapes, empty bins behavior)
- Metrics correctness and invariants (mean_shift row sum ≈ 0, lift definition sanity)
- Missingness/edge-case behavior (per-class totals and missing pct)
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
import pytest
from attrs import frozen

from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.core.types import float64, int64, string
from agentune.analyze.feature.base import BoolFeature, CategoricalFeature, Feature, FloatFeature
from agentune.analyze.feature.problem import (
    ClassificationProblem,
    ProblemDescription,
    RegressionProblem,
)
from agentune.analyze.feature.stats.stats_calculators import UnifiedRelationshipStatsCalculator
from agentune.analyze.join.base import JoinStrategy


# Helper functions for creating Problem instances concisely
def _classification_problem(target_series: pl.Series) -> ClassificationProblem:
    """Create a ClassificationProblem for testing with string target."""
    return ClassificationProblem(
        problem_description=ProblemDescription('y', problem_type='classification'),
        target_column=Field('y', string),
        classes=tuple(sorted(target_series.drop_nulls().unique().to_list()))
    )

def _regression_problem() -> RegressionProblem:
    """Create a RegressionProblem for testing with float target."""
    return RegressionProblem(
        problem_description=ProblemDescription('y', problem_type='regression'),
        target_column=Field('y', float64)
    )


# Simple Feature implementations for testing (consistent with other stats tests)
@frozen
class SimpleBoolFeature(BoolFeature):
    name: str
    description: str = 'Test bool feature'
    code: str = 'def evaluate(df): return df[self.name]'

    # Redeclare attributes with defaults
    default_for_missing: bool = False

    @property
    def params(self) -> Schema:
        return Schema((Field(self.name, self.dtype),))

    @property
    def secondary_tables(self) -> Sequence[DuckdbTable]:
        return []

    @property
    def join_strategies(self) -> Sequence[JoinStrategy]:
        return []


@frozen
class SimpleNumericFeature(FloatFeature):
    name: str
    description: str = 'Test numeric feature'
    code: str = 'def evaluate(df): return df[self.name]'
    technical_description = 'A simple numeric feature'

    default_for_missing: float = 0.0
    default_for_nan: float = 0.0
    default_for_infinity: float = 0.0
    default_for_neg_infinity: float = 0.0

    @property
    def params(self) -> Schema:
        return Schema((Field(self.name, self.dtype),))

    @property
    def secondary_tables(self) -> Sequence[DuckdbTable]:
        return []

    @property
    def join_strategies(self) -> Sequence[JoinStrategy]:
        return []


@frozen
class SimpleCategoricalFeature(CategoricalFeature):
    name: str
    categories: tuple[str, ...]
    description: str = 'Test categorical feature'
    code: str = 'def evaluate(df): return df[self.name]'
    technical_description = 'A simple categorical feature'

    default_for_missing: str = CategoricalFeature.other_category

    @property
    def params(self) -> Schema:
        return Schema((Field(self.name, self.dtype),))

    @property
    def secondary_tables(self) -> Sequence[DuckdbTable]:
        return []

    @property
    def join_strategies(self) -> Sequence[JoinStrategy]:
        return []


# ------------------------------
# A. Type coverage smoke tests
# ------------------------------
@pytest.mark.parametrize(
    ('feature_kind', 'target_kind'),
    [
        ('bool', 'categorical'),
        ('bool', 'numeric'),
        ('numeric', 'categorical'),
        ('numeric', 'numeric'),
        ('categorical', 'categorical'),
        ('categorical', 'numeric'),
    ],
)
def test_unified_type_coverage(feature_kind: str, target_kind: str) -> None:
    feature: Feature
    # Build feature series and feature object
    if feature_kind == 'bool':
        feat_series = pl.Series('f', [True, False, True, None, True, False, True, False])
        feature = SimpleBoolFeature(name='f', technical_description='A bool feature')
        expected_rows = 2  # "False", "True"
    elif feature_kind == 'numeric':
        feat_series = pl.Series('f', [1.0, 2.0, 3.0, 4.0, None, 5.0, 6.0, 7.0])
        feature = SimpleNumericFeature(name='f', technical_description='A numeric feature')
        expected_rows = 5  # default numeric_feature_bins
    else:  # categorical
        feat_series = pl.Series('f', ['A', 'B', 'A', None, 'C', 'A', 'B', 'C'])
        feature = SimpleCategoricalFeature(name='f', categories=('A', 'B', 'C'), technical_description='A categorical feature')
        expected_rows = 3

    # Build target series
    if target_kind == 'categorical':
        target_series = pl.Series('y', ['X', 'Y', 'X', 'Y', 'Z', 'Z', 'X', 'Y'])
        expected_classes = ('X', 'Y', 'Z')
    else:
        target_series = pl.Series('y', [1.0, 2.0, 1.5, 2.5, 3.0, 3.5, 0.5, 4.0])
        expected_classes = None  # numeric target is binned

    calc = UnifiedRelationshipStatsCalculator()
    problem = _classification_problem(target_series) if target_kind == 'categorical' else _regression_problem()
    stats = calc.calculate_from_series(feature, feat_series, target_series, problem)

    # Basic assertions
    assert stats.n_total == feat_series.len()
    assert stats.n_missing == feat_series.null_count()

    # Classes present and sorted
    assert isinstance(stats.classes, tuple)
    if target_kind == 'categorical':
        assert stats.classes == expected_classes
    else:
        # default 5 bins for numeric targets; ensure there is the right number of bins
        # Some bins may be missing with tiny data, but labels should all start with TQ
        assert len(stats.classes) <= 6

    # Shapes of matrices: rows = expected_rows, cols = len(classes)
    k = expected_rows
    c = len(stats.classes)
    assert stats.lift is not None and stats.mean_shift is not None
    assert len(stats.lift) == k and len(stats.mean_shift) == k
    assert all(len(row) == c for row in stats.lift)
    assert all(len(row) == c for row in stats.mean_shift)


# -----------------------------------
# B. Binning behavior and edge cases
# -----------------------------------

def test_numeric_feature_binning_rows_and_constant_feature() -> None:
    # Non-constant numeric feature with explicit bins
    feat_series = pl.Series('f', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    target = pl.Series('y', ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B'])  # categorical
    feature = SimpleNumericFeature(name='f', technical_description='A numeric feature')

    calc = UnifiedRelationshipStatsCalculator(numeric_feature_bins=4)
    problem = _classification_problem(target)
    stats = calc.calculate_from_series(feature, feat_series, target, problem)

    # Expect 4 rows (Q1..Q4)
    assert stats.lift is not None and len(stats.lift) == 4
    assert stats.mean_shift is not None and len(stats.mean_shift) == 4

    # Constant numeric feature → qcut naturally produces only one bin
    const_series = pl.Series('f', [5.0] * 12)
    target_const = pl.Series('y', ['A', 'B'] * 6)
    problem_const = _classification_problem(target_const)
    stats_const = calc.calculate_from_series(feature, const_series, target_const, problem_const)
    assert stats_const.lift is not None and len(stats_const.lift) == 1
    assert stats_const.mean_shift is not None and len(stats_const.mean_shift) == 1


def test_numeric_target_binning_labels_and_constant_target() -> None:
    feature = SimpleCategoricalFeature(name='f', categories=('X', 'Y'), technical_description='A categorical feature')
    feat_series = pl.Series('f', ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'])  # balanced

    # Non-constant numeric target with 4 bins
    target = pl.Series('y', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    calc = UnifiedRelationshipStatsCalculator(target_numeric_bins=4)
    problem = _regression_problem()
    stats = calc.calculate_from_series(feature, feat_series, target, problem)

    # Expect 4 binned class labels should have a length of 4
    assert len(stats.classes) == 4

    # Constant numeric target → single class
    target_const = pl.Series('y', [1.0] * 8)
    problem_const = _regression_problem()
    stats_const = calc.calculate_from_series(feature, feat_series, target_const, problem_const)
    assert len(stats_const.classes) == 1


# -----------------------------------
# C. Metrics correctness & invariants
# -----------------------------------

def test_mean_shift_row_sums_zero() -> None:
    # Use a simple categorical feature and target
    feature = SimpleCategoricalFeature(name='f', categories=('X', 'Y', 'Z'), technical_description='A categorical feature')
    feat_series = pl.Series('f', ['X', 'X', 'Y', 'Y', 'Z', 'Z', 'X', 'Y', None])
    target = pl.Series('y', ['A', 'B', 'A', 'B', 'A', 'B', 'B', 'A', 'A'])  # 8 valid

    calc = UnifiedRelationshipStatsCalculator()
    problem = _classification_problem(target)
    stats = calc.calculate_from_series(feature, feat_series, target, problem)

    # For each row (category), sum of mean_shift over classes should be ~ 0
    assert stats.mean_shift is not None
    for row in stats.mean_shift:
        finite_vals = [v for v in row if not np.isnan(v)]
        if finite_vals:  # skip empty-bin rows which are all NaN
            assert sum(finite_vals) == pytest.approx(0.0, abs=1e-12)


def test_lift_definition_small_dataset() -> None:
    # Handcrafted 2x2 perfect separation for clarity
    feature = SimpleCategoricalFeature(name='f', categories=('X', 'Y'), technical_description='A categorical feature')
    feat_series = pl.Series('f', ['X', 'X', 'Y', 'Y'])
    target = pl.Series('y', ['A', 'A', 'B', 'B'])  # P(A)=P(B)=0.5

    calc = UnifiedRelationshipStatsCalculator()
    problem = _classification_problem(target)
    stats = calc.calculate_from_series(feature, feat_series, target, problem)

    # Lift should be: for X → [2.0, 0.0]; for Y → [0.0, 2.0]
    assert stats.lift is not None
    x_row, y_row = stats.lift
    # Order of classes is sorted: ("A", "B")
    assert stats.classes == ('A', 'B')
    assert x_row[0] == pytest.approx(2.0, abs=1e-12)
    assert x_row[1] == pytest.approx(0.0, abs=1e-12)
    assert y_row[0] == pytest.approx(0.0, abs=1e-12)
    assert y_row[1] == pytest.approx(2.0, abs=1e-12)


def test_independence_info_gain_near_zero_and_lift_one() -> None:
    # Independent feature and target: each category has same class distribution
    feature = SimpleCategoricalFeature(name='f', categories=('X', 'Y'), technical_description='A categorical feature')
    # 50 X and 50 Y
    feat_series = pl.Series('f', ['X'] * 50 + ['Y'] * 50)
    # Within each, 25 A and 25 B
    target = pl.Series('y', ['A'] * 25 + ['B'] * 25 + ['A'] * 25 + ['B'] * 25)

    calc = UnifiedRelationshipStatsCalculator()
    problem = _classification_problem(target)
    stats = calc.calculate_from_series(feature, feat_series, target, problem)

    assert stats.lift is not None
    assert stats.mean_shift is not None
    for row in stats.lift:
        for val in row:
            assert val == pytest.approx(1.0, abs=1e-12)
    for row in stats.mean_shift:
        for val in row:
            assert val == pytest.approx(0.0, abs=1e-12)


# -----------------------------------
# D. Missingness and per-class totals
# -----------------------------------

def test_missingness_per_class_and_totals() -> None:
    # Test that missing percentages and totals are calculated correctly
    feature = SimpleCategoricalFeature(name='f', categories=('X', 'Y'), technical_description='A categorical feature')
    # Feature: X, Y, X, None, Y, X (missing in position 3)
    feat_series = pl.Series('f', ['X', 'Y', 'X', None, 'Y', 'X'])
    # Target: A, A, B, B, B, A (classes A=3, B=3)
    target = pl.Series('y', ['A', 'A', 'B', 'B', 'B', 'A'])

    calc = UnifiedRelationshipStatsCalculator()
    problem = _classification_problem(target)
    stats = calc.calculate_from_series(feature, feat_series, target, problem)

    # Check totals per class
    assert stats.totals_per_class == (3, 3)  # 3 A's, 3 B's

    # Check missing percentages per class
    # Class A: positions 0, 1, 5 → missing at position 3? No, position 3 has target B
    # Class B: positions 2, 3, 4 → missing at position 3? Yes, feature is None
    # So: A has 0/3 missing, B has 1/3 missing
    expected_missing_pct = (0.0, 1.0/3.0)
    assert len(stats.missing_percentage_per_class) == 2
    assert stats.missing_percentage_per_class[0] == pytest.approx(expected_missing_pct[0], abs=1e-12)
    assert stats.missing_percentage_per_class[1] == pytest.approx(expected_missing_pct[1], abs=1e-12)


def test_binning_info_for_numeric_features() -> None:
    """Test that BinningInfo is populated correctly for numeric features."""
    # Test numeric feature
    feature = SimpleNumericFeature(name='f', technical_description='A numeric feature')
    feat_series = pl.Series('f', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    target = pl.Series('y', ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'])

    calc = UnifiedRelationshipStatsCalculator(numeric_feature_bins=4)
    problem = _classification_problem(target)
    stats = calc.calculate_from_series(feature, feat_series, target, problem)

    # Check that binning_info is present for numeric features
    assert stats.binning_info is not None
    assert stats.binning_info.n_bins() == 4
    assert len(stats.binning_info.bin_edges) == 5  # n_bins + 1 edges
    
    # Check that edges use -inf and +inf for boundaries
    assert stats.binning_info.bin_edges[0] == float('-inf')
    assert stats.binning_info.bin_edges[-1] == float('inf')
    
    # Check that internal edges are reasonable (between min and max values)
    internal_edges = stats.binning_info.bin_edges[1:-1]
    for edge in internal_edges:
        assert 1.0 <= edge <= 8.0


def test_binning_info_none_for_categorical_features() -> None:
    """Test that BinningInfo is None for categorical features."""
    # Test categorical feature
    feature = SimpleCategoricalFeature(name='f', categories=('X', 'Y', 'Z'), technical_description='A categorical feature')
    feat_series = pl.Series('f', ['X', 'Y', 'Z', 'X', 'Y', 'Z'])
    target = pl.Series('y', ['A', 'B', 'A', 'B', 'A', 'B'])

    calc = UnifiedRelationshipStatsCalculator()
    problem = _classification_problem(target)
    stats = calc.calculate_from_series(feature, feat_series, target, problem)

    # Check that binning_info is None for categorical features
    assert stats.binning_info is None


def test_binned_series_matches_bin_edges() -> None:
    """Test that binned series values correspond correctly to bin edges boundaries."""
    from agentune.analyze.feature.stats.stats_calculators import _bin_numeric_to_quantiles
    
    # Test with a well-distributed series
    series = pl.Series('f', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    binned_series, bin_edges = _bin_numeric_to_quantiles(series, 4)
    
    # Verify bin_edges structure: should have -inf, internal edges, +inf
    assert bin_edges[0] == float('-inf')
    assert bin_edges[-1] == float('inf')
    assert len(bin_edges) == 5  # 4 bins = 5 edges
    
    # Verify internal edges are in ascending order
    internal_edges = bin_edges[1:-1]
    assert all(internal_edges[i] <= internal_edges[i+1] for i in range(len(internal_edges)-1))
    
    # Verify all binned values are within expected ranges
    unique_bins = binned_series.drop_nulls().unique().sort().to_list()
    assert len(unique_bins) == 4  # Should have exactly 4 bins
    
    # Check that bin labels reflect the -inf/+inf boundaries
    first_bin = unique_bins[0]
    last_bin = unique_bins[-1]
    assert '(-inf,' in first_bin
    assert 'inf]' in last_bin
    
    # Test with constant series
    const_series = pl.Series('f', [5.0] * 8)
    binned_const, bin_edges_const = _bin_numeric_to_quantiles(const_series, 4)
    
    # Constant series should still have proper bin_edges structure
    assert bin_edges_const[0] == float('-inf')
    assert bin_edges_const[-1] == float('inf')
    
    # Should produce only 1 unique bin for constant values
    unique_const_bins = binned_const.drop_nulls().unique().to_list()
    assert len(unique_const_bins) == 1


def test_class_ordering_follows_problem_definition() -> None:
    """Test that classes appear in the order defined in the Problem, not data order.
    
    This test verifies that:
    1. For classification with non-numeric targets: Classes appear in the official order from the Problem object
    2. Classes not in data still appear with 0/NaN values as appropriate
    3. Classes in data but not in official list appear after official classes
    4. For regression or classification with numeric targets: Uses data ordering (since classes come from binning)
    """
    # Create a problem with specific class ordering: ['A', 'B', 'C'] (must be sorted)
    problem_desc = ProblemDescription(target_column='target')
    target_field = Field(name='target', dtype=string)
    problem = ClassificationProblem(
        problem_description=problem_desc,
        target_column=target_field,
        classes=('A', 'B', 'C')  # Official ordering: A, B, C (sorted)
    )
    
    # Create feature
    feature = SimpleCategoricalFeature(name='feature', categories=('X', 'Y'), technical_description='Test feature')
    
    # Create data where classes appear in different order and some official classes are missing
    # Data has classes in order: A, B, D (missing C, extra D)
    feature_data = pl.Series('feature', ['X', 'X', 'Y', 'Y', 'X'])
    target_data = pl.Series('target', ['A', 'B', 'A', 'D', 'B'])  # D is not in official classes
    
    # Calculate stats
    calc = UnifiedRelationshipStatsCalculator()
    stats = calc.calculate_from_series(feature, feature_data, target_data, problem)
    
    # Verify the ordering: official classes first, then extra classes
    expected_classes = ('A', 'B', 'C', 'D')  # Official classes first, then extra classes
    assert stats.classes == expected_classes, f'Expected {expected_classes}, got {stats.classes}'
    
    # Verify totals match the ordering
    # A: 2, B: 2, C: 0 (not in data), D: 1
    expected_totals = (2, 2, 0, 1)
    assert stats.totals_per_class == expected_totals, f'Expected {expected_totals}, got {stats.totals_per_class}'
    
    # Verify missing percentages match the ordering (all should be 0.0 since no feature values are missing)
    expected_missing_pct = (0.0, 0.0, float('nan'), 0.0)  # NaN for class C which has 0 total
    assert len(stats.missing_percentage_per_class) == 4
    assert stats.missing_percentage_per_class[0] == pytest.approx(expected_missing_pct[0], abs=1e-12)
    assert stats.missing_percentage_per_class[1] == pytest.approx(expected_missing_pct[1], abs=1e-12)
    assert np.isnan(stats.missing_percentage_per_class[2])  # NaN for empty class
    assert stats.missing_percentage_per_class[3] == pytest.approx(expected_missing_pct[3], abs=1e-12)
    
    # Verify that the confusion matrix dimensions match
    assert stats.lift is not None and stats.mean_shift is not None
    assert len(stats.lift) == 2  # 2 feature categories (X, Y)
    assert len(stats.mean_shift) == 2  # 2 feature categories (X, Y)
    assert all(len(row) == 4 for row in stats.lift)  # 4 classes (A, B, C, D)
    assert all(len(row) == 4 for row in stats.mean_shift)  # 4 classes (A, B, C, D)


def test_numeric_classification_not_rebinned() -> None:
    """Test that numeric classification targets are not re-binned.
    
    This verifies that when we have a classification problem with numeric classes,
    the target values are treated as categorical labels (not binned into quantiles).
    """
    # Create a numeric classification problem with classes [0, 1, 2]
    problem_desc = ProblemDescription(target_column='target', problem_type='classification')
    target_field = Field(name='target', dtype=int64)
    classification_problem = ClassificationProblem(
        problem_description=problem_desc,
        target_column=target_field,
        classes=(0, 1, 2)  # Numeric classes but still classification
    )
    
    # Create a regression problem for comparison
    regression_desc = ProblemDescription(target_column='target', problem_type='regression')
    regression_problem = RegressionProblem(
        problem_description=regression_desc,
        target_column=target_field
    )
    
    # Create test data
    feature = SimpleNumericFeature(name='test_feature', technical_description='Test feature')
    feature_series = pl.Series([1, 2, 3, 4, 5, 6])
    target_series = pl.Series([0, 1, 2, 0, 1, 2])  # Numeric target with classes 0, 1, 2
    
    # Create calculator
    calc = UnifiedRelationshipStatsCalculator(target_numeric_bins=3)
    
    # Test classification: should NOT re-bin the target
    classification_stats = calc.calculate_from_series(feature, feature_series, target_series, classification_problem)
    
    # Test regression: should bin the target
    regression_stats = calc.calculate_from_series(feature, feature_series, target_series, regression_problem)
    
    # Verify results
    # For classification: classes should be ['0', '1', '2'] (original classes as strings)
    expected_classification_classes = ('0', '1', '2')
    assert classification_stats.classes == expected_classification_classes, \
        f'Classification classes should be {expected_classification_classes}, got {classification_stats.classes}'
    
    # For regression: classes should be binned intervals (different from original values)
    # The exact values depend on quantile binning, but should be different from ['0', '1', '2']
    assert regression_stats.classes != expected_classification_classes, \
        f'Regression classes should be binned (different from original), got {regression_stats.classes}'
    
    # Verify that classification preserves the class counts correctly
    # We have 2 instances each of classes 0, 1, 2
    expected_totals = (2, 2, 2)
    assert classification_stats.totals_per_class == expected_totals, \
        f'Expected totals {expected_totals}, got {classification_stats.totals_per_class}'
