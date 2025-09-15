from __future__ import annotations

from collections.abc import Sequence
from typing import cast, override

import numpy as np
import polars as pl

from agentune.analyze.feature.base import (
    BoolFeature,
    CategoricalFeature,
    Feature,
    NumericFeature,
)
from agentune.analyze.feature.problem import Problem
from agentune.analyze.feature.stats.base import (
    BinningInfo,
    CategoryClassMatrix,
    FeatureStats,
    RelationshipStats,
    SyncFeatureStatsCalculator,
    SyncRelationshipStatsCalculator,
)
from agentune.analyze.util.feature_sse_reduction import calculate_sse_reduction

# ---------------------------------------------------------------------------
# Helper functions for unified assembly
# ---------------------------------------------------------------------------


def _mean_shift_and_lift_from_confusion(matrix: np.ndarray
                                        ) -> tuple[CategoryClassMatrix, CategoryClassMatrix]:
    """Compute mean shift and lift from a confusion matrix."""
    n = int(matrix.sum())
    p_class = matrix.sum(axis=0) / n  # (c,)  The overall class probabilities
    row_sums = matrix.sum(axis=1)  # (k,)  The counts per category
    mean_shift_rows: list[tuple[float, ...]] = []
    lift_rows: list[tuple[float, ...]] = []
    for i in range(matrix.shape[0]):
        denom = row_sums[i]
        if denom > 0:
            p_class_given_cat = matrix[i, :] / denom
            mean_shift_rows.append(tuple(float(x) for x in (p_class_given_cat - p_class).tolist()))
            row_lift = [
                (p_class_given_cat[j] / p_class[j]) if p_class[j] > 0 else float('nan')
                for j in range(matrix.shape[1])
            ]
            lift_rows.append(tuple(float(x) for x in row_lift))
        else:
            raise ValueError('An empty row was encountered in the confusion matrix.')
    return tuple(mean_shift_rows), tuple(lift_rows)


# ---------------------------------------------------------------------------
# Helper functions for unified-from-data builder
# ---------------------------------------------------------------------------
def _compute_support_categorical(series: pl.Series, categories: tuple[str, ...]) -> tuple[float, ...]:
    non_null = series.drop_nulls()
    counts: dict[str, int] = dict(non_null.value_counts().rows())
    denom = non_null.len() or 1
    return tuple(counts.get(cat, 0) / denom for cat in categories)


def _bin_numeric_to_quantiles(series: pl.Series, k: int
                              ) -> tuple[pl.Series, tuple[float, ...]]:
    """Bin a numeric series into k quantile-based categories using Polars' qcut.

    Returns a tuple of (label_series, bin_edges_tuple).
    Bin edges use -inf and +inf for the first and last edges to handle all possible values.
    """
    non_null = series.drop_nulls()
    if non_null.len() == 0:
        bin_edges: tuple[float, ...] = (float('-inf'), float('inf'))
        return series, bin_edges

    # Use qcut on full series to preserve length and nulls
    binned_series = series.qcut(k, allow_duplicates=True)
    
    # Calculate bin edges with -inf and +inf boundaries
    quantiles = [i / k for i in range(1, k)]
    internal_edges = [float(cast(float, non_null.quantile(q, interpolation='linear'))) for q in quantiles]
    bin_edges = (float('-inf'), *internal_edges, float('inf'))

    return binned_series, bin_edges


def _build_confusion_matrix(cats: Sequence[object], classes: Sequence[object], df: pl.DataFrame
                            ) -> np.ndarray:
    """Build a confusion matrix from the provided DataFrame with 'cat' and 'class' columns."""
    counts = df.group_by(['cat', 'class']).len().sort(['cat', 'class']).iter_rows(named=True)
    cats_list = list(cats)
    classes_list = list(classes)
    k, c = len(cats_list), len(classes_list)
    matrix = np.zeros((k, c), dtype=int)
    for row in counts:
        cat = row['cat']
        cls = row['class']
        if cat in cats_list and cls in classes_list:
            i = cats_list.index(cat)
            j = classes_list.index(cls)
            matrix[i, j] = row['len']
    return matrix


def _missing_and_totals_per_class(feature_series: pl.Series, class_series: pl.Series
                                  ) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Compute missing percentage and totals per class using the provided class labels.
    class_series should already reflect any binning that was applied to the target.
    """
    classes = class_series.drop_nulls().unique().sort().to_list()
    missing_pct: list[float] = []
    totals: list[int] = []
    for cls in classes:
        cls_mask = (class_series == cls).fill_null(False)
        cls_total = int(cls_mask.sum())
        totals.append(cls_total)
        if cls_total == 0:
            missing_pct.append(float('nan'))
        else:
            miss_count = int(feature_series.filter(cls_mask).is_null().sum())
            missing_pct.append(miss_count / cls_total)
    return tuple(missing_pct), tuple(totals)


# Helper functions required by unified builder
def _feature_to_categorical(feature: Feature, series: pl.Series, numeric_feature_bins: int = 5,
                            ) -> tuple[pl.Series, tuple[str, ...], tuple[float, ...] | None]:
    """Return a categorical view of any supported feature type.

    - CategoricalFeature → keep values as strings; labels from feature.categories
    - BoolFeature → map to "False"/"True"; labels ("False", "True")
    - NumericFeature → quantile bin into labels Q1..Qk

    Returns:
        tuple of (categorical_series, labels, bin_edges_or_none)
        bin_edges_or_none is None for non-numeric features, tuple of bin edges for numeric features
    """
    if isinstance(feature, CategoricalFeature):
        labels = tuple(str(c) for c in feature.categories)
        # Convert values to strings, preserving nulls
        cat = series.cast(pl.Utf8)
        return cat, labels, None
    if isinstance(feature, BoolFeature):
        raw = series.to_list()
        out = [None if v is None else ('True' if bool(v) else 'False') for v in raw]
        labels = ('False', 'True')
        return pl.Series(out, dtype=pl.Utf8), labels, None
    if isinstance(feature, NumericFeature):
        binned, bin_edges = _bin_numeric_to_quantiles(series, numeric_feature_bins)
        actual_categories = binned.drop_nulls().unique().sort().to_list()
        labels = tuple(str(cat) for cat in actual_categories)
        return binned, labels, bin_edges
    # With strict assumptions, other types are not expected
    raise TypeError(f'Unsupported feature type for unified stats: {type(feature)}')


def _prepare_class_target_series(target_series: pl.Series, bins: int, is_numeric: bool = False) -> pl.Series:
    """Prepare target series for classification metrics.

    If numeric → bin to quantiles with quantile names being the range. Otherwise cast to string labels.
    """
    if is_numeric:
        labels_series, _ = _bin_numeric_to_quantiles(target_series, bins)
        return labels_series
    else:
        return target_series.cast(pl.Utf8)


# ---------------------------------------------------------------------------
# Unified builder directly from raw data (no wrapping of concrete stats)
# ---------------------------------------------------------------------------
class UnifiedStatsCalculator(SyncFeatureStatsCalculator):
    """Unified stats calculator from raw data (no wrapping of concrete stats).
    
    Args:
        numeric_feature_bins: The number of quantile-based bins to use when
            converting a numeric feature into a categorical representation.
    """

    def __init__(self, numeric_feature_bins: int = 5) -> None:
        self.numeric_feature_bins = numeric_feature_bins

    @override
    def calculate_from_series(self, feature: Feature, series: pl.Series) -> FeatureStats:
        """Compute unified feature statistics for any feature type.

        Args:
            feature: The feature to calculate statistics for
            series: A polars Series containing the feature data

        Returns:
            Unified feature statistics

        """
        n_total = int(series.len())
        n_missing = int(series.null_count())
        # Turn the feature into a categorical view for unified stats
        # CategoricalFeature: keep as-is; BoolFeature: map to "False"/"True"; NumericFeature: quantile bin
        cat_series_full, cat_labels, _ = _feature_to_categorical(feature, series, numeric_feature_bins=self.numeric_feature_bins)

        non_null_series = cat_series_full.drop_nulls()
        counts_dict: dict[str, int] = dict(non_null_series.value_counts().rows())
        value_counts = {str(k): counts_dict.get(str(k), 0) for k in cat_labels}

        # Calculate support (per-category percentages)
        support = _compute_support_categorical(series, cat_labels)

        return FeatureStats(
            n_total=n_total,
            n_missing=n_missing,
            categories=tuple(str(cat) for cat in cat_labels),
            value_counts=value_counts,
            support=support,
        )


class UnifiedRelationshipStatsCalculator(SyncRelationshipStatsCalculator):
    """Unified relationship calculator that treats all tasks as classification.

    This calculator provides a consistent way to analyze the relationship between
    any feature and a target variable by treating every analysis as a
    classification problem.

    It works by first converting the input feature into a categorical
    representation (e.g., by binning numeric features). It then does the same
    for the target: if the target is numeric, it is binned into quantiles to
    create discrete classes. With both feature and target in a categorical
    format, it computes a standard set of classification statistics, such as
    lift, information gain, and probability shift.

    Args:
        numeric_feature_bins: The number of quantile-based bins to use when
            converting a numeric feature into a categorical representation.
        target_numeric_bins: The number of quantile-based bins to use when
            converting a numeric target into a categorical representation.
    """

    def __init__(self, numeric_feature_bins: int = 5, target_numeric_bins: int = 5) -> None:
        self.numeric_feature_bins = numeric_feature_bins
        self.target_numeric_bins = target_numeric_bins


    @override
    def calculate_from_series(self, feature: Feature, series: pl.Series, target: pl.Series, problem: Problem
                              ) -> RelationshipStats:
        # 1. Unify the Feature into a categorical series
        cat_series_full, cat_labels, bin_edges = _feature_to_categorical(
            feature, series, numeric_feature_bins=self.numeric_feature_bins
        )
        cats = list(cat_labels)

        # 2. Unify the Target into a categorical "class" series
        # Determine if the target is numeric and needs binning.
        is_numeric_target = target.dtype.is_numeric()
        class_series_full = _prepare_class_target_series(
            target, bins=self.target_numeric_bins, is_numeric=is_numeric_target
        )

        # 3. Perform Universal Classification Metric Calculations
        df_full = pl.DataFrame({'cat': cat_series_full, 'class': class_series_full})
        df_non_null = df_full.drop_nulls(['cat', 'class'])

        classes_list = class_series_full.drop_nulls().unique().sort().to_list()
        classes = tuple(str(c) for c in classes_list)

        # Build the confusion matrix, which is the basis for most metrics
        matrix = _build_confusion_matrix(cats, classes_list, df_non_null)

        # Calculate probability shift and lift from the confusion matrix
        prob_shift, lift = _mean_shift_and_lift_from_confusion(matrix)

        # Calculate per-class totals and missing percentages
        missing_pct, totals = _missing_and_totals_per_class(series, class_series_full)

        # Calculate SSE reduction for sse_reduction field
        sse_reduction = calculate_sse_reduction(feature, series, target)

        # Create binning info if bin_edges is available (numeric features only)
        binning_info = None
        if bin_edges is not None:
            binning_info = BinningInfo(
                bin_edges=bin_edges
            )

        # 4. Assemble the final stats object
        return RelationshipStats(
            n_total=int(series.len()),
            n_missing=int(series.null_count()),
            classes=classes,
            # The 'mean_shift' field now always stores the probability shift
            mean_shift=prob_shift,
            lift=lift,
            sse_reduction=sse_reduction,
            binning_info=binning_info,
            totals_per_class=totals,
            missing_percentage_per_class=missing_pct,
        )
