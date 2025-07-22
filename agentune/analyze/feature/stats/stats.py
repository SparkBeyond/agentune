from __future__ import annotations

from attrs import frozen
from frozendict import frozendict

from agentune.analyze.feature.base import (
    BoolFeature,
    CategoricalFeature,
    Classification,
    NumericFeature,
    Regression,
)
from agentune.analyze.feature.stats.base import FeatureStats, RelationshipStats


# --- Concrete feature-stats subclasses ---
@frozen
class BooleanFeatureStats(FeatureStats[BoolFeature]):
    n_total: int
    n_missing: int
    support: int  # Number of True values
    coverage: float  # Fraction of non-null values that are True


@frozen
class NumericFeatureStats(FeatureStats[NumericFeature]):
    n_total: int
    n_missing: int
    min: float
    max: float
    mean: float
    std: float
    variance: float
    median: float
    q1: float
    q3: float


@frozen
class CategoricalFeatureStats(FeatureStats[CategoricalFeature]):
    n_total: int
    n_missing: int
    unique_count: int
    value_counts: frozendict[str, int]
    # entropy


# --- Concrete relationship-stats subclasses and calculators ---
@frozen
class BooleanRegressionStats(RelationshipStats[BoolFeature, Regression]):
    n_samples: int
    n_missing_feature: int
    mean_true: float  # Mean target value when feature is True
    mean_false: float  # Mean target value when feature is False
    shift: float  # Difference between mean_true and overall mean
    point_biserial: float  # Point-biserial correlation coefficient


@frozen
class BooleanClassificationStats(RelationshipStats[BoolFeature, Classification]):
    n_samples: int
    n_missing_feature: int
    true_counts: tuple[int, ...]  # Count of True values for each target category
    categories: tuple[str, ...]  # The target categories
    info_gain: float  # Mutual information between feature and target
    lift: tuple[float, ...]  # P(class|True) / P(class) for each category
    shift: tuple[float, ...]  # P(class|True) - P(class) for each category


@frozen
class NumericRegressionStats(RelationshipStats[NumericFeature, Regression]):
    n_samples: int
    n_missing_feature: int
    pearson_r: float  # Pearson correlation coefficient
    pearson_p: float  # P-value for Pearson correlation
    spearman_r: float  # Spearman rank correlation coefficient
    spearman_p: float  # P-value for Spearman correlation


@frozen
class NumericClassificationStats(RelationshipStats[NumericFeature, Classification]):
    n_samples: int
    n_missing_feature: int
    anova_f: float
    p_value: float


@frozen
class CategoricalRegressionStats(RelationshipStats[CategoricalFeature, Regression]):
    n_samples: int
    n_missing_feature: int
    category_counts: tuple[int, ...]  # Count of samples in each category
    categories: tuple[str, ...]  # The feature categories
    mean_by_category: tuple[float, ...]  # Mean target value for each category
    shift_by_category: tuple[float, ...]  # Difference between category mean and overall mean


@frozen
class CategoricalClassificationStats(RelationshipStats[CategoricalFeature, Classification]):
    n_samples: int
    n_missing_feature: int
    categories: tuple[str, ...]  # The feature categories
    classes: tuple[str, ...]  # The target classes
    category_class_counts: tuple[tuple[int, ...], ...]
    # Contingency matrix counts: category_class_counts[i][j] is the count of samples with category categories[i] and class classes[j].
    # The outer tuple is per category (feature value), the inner tuple is per class (target value).
    #
    # Example:
    #   categories = ("red", "blue")
    #   classes = ("A", "B", "C")
    #   category_class_counts = (
    #       (3, 1, 2),  # "red": 3 samples in class "A", 1 in "B", 2 in "C"
    #       (0, 4, 5),  # "blue": 0 in "A", 4 in "B", 5 in "C"
    #   )
    info_gain: float  # Mutual information between feature and target
    lift: tuple[tuple[float, ...], ...]  # P(class|category) / P(class) for each combination
    shift: tuple[tuple[float, ...], ...]  # P(class|category) - P(class) for each combination
