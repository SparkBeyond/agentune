"""Joint utility for single feature SSE reduction calculations.

This module provides shared functionality for calculating Sum of Squared Errors (SSE)
reduction when evaluating individual features against targets. It supports both
regression and classification tasks with proper target encoding.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from attrs import frozen

from agentune.analyze.feature.base import (
    CategoricalFeature,
    Feature,
)

# ---------------------------------------------------------------------------
# SSE calculation data structures
# ---------------------------------------------------------------------------

@frozen
class TargetStats:
    """Baseline target statistics for SSE calculations."""
    sy: np.ndarray
    sy2: np.ndarray
    priorsses: np.ndarray
    stdevs: np.ndarray
    n_samples: float


@frozen
class FeatureTargetStats:
    """Per-feature joint statistics against target(s)."""
    sx: np.ndarray
    sx2: np.ndarray
    sxy: np.ndarray
    sses: np.ndarray


# ---------------------------------------------------------------------------
# Core SSE calculation functions
# ---------------------------------------------------------------------------

def solve_2x2_system(a11: float, a12: float, a21: float, a22: float, b1: float, b2: float) -> tuple:
    """Solve 2x2 linear system for single-variable regression with intercept.
    Returns (a, c) for y = a*x + c.
    """
    matrix_a = np.array([[a11, a12], [a21, a22]])
    b = np.array([b1, b2])
    try:
        solution = np.linalg.solve(matrix_a, b)
        return tuple(solution)
    except np.linalg.LinAlgError:
        return (0.0, 0.0)


def lin_regression_1variable_with_sums(sx: float, sy: float, sx2: float, sxy: float, n: float, sy2: float) -> float:
    """Closed-form single-variable linear regression with intercept using sums.
    Returns SSE.
    """
    a, c = solve_2x2_system(sx2, sx, sx, n, sxy, sy)
    e = (a * a * sx2 + 2 * a * c * sx - 2 * a * sxy + n * c * c - 2 * c * sy + sy2)
    return e


def calculate_baseline_statistics(target: np.ndarray) -> TargetStats:
    """Calculate baseline target statistics for SSE reduction calculations.
    
    Args:
        target: Target values as a 2D numpy array with shape (n_samples, n_targets).
                For single-target problems, this is (n_samples, 1).
                For multiclass classification problems that have been one-hot encoded,
                this is (n_samples, n_classes) where each column represents one class.
                The array format is required to handle both regression and multiclass
                scenarios uniformly using vectorized operations.
    
    Returns:
        TargetStats containing baseline statistics (sums, sum of squares, prior SSEs,
        standard deviations) needed for calculating SSE reduction when features are added.
    """
    n_samples = float(target.shape[0])
    sy = target.sum(axis=0)
    sy2 = (target * target).sum(axis=0)
    prior_sses = sy2 - (sy * sy / n_samples)
    stdevs = np.sqrt(prior_sses / n_samples)
    return TargetStats(sy=sy, sy2=sy2, priorsses=prior_sses, stdevs=stdevs, n_samples=n_samples)


def calculate_feature_statistics(feature_values: np.ndarray, target: np.ndarray) -> FeatureTargetStats:
    """Calculate per-feature joint sums and SSEs against target(s)."""
    n_targets = target.shape[1]
    n_samples = float(target.shape[0])

    if not np.all(np.isfinite(feature_values)):
        raise ValueError('an infinite value was passed')

    # Normalize features to shape (n_samples, n_targets):
    # - 1D or single-column features are reused across all targets
    # - Multi-component features must have exactly one component per target
    x = feature_values[:, None] if feature_values.ndim == 1 else feature_values
    x = np.repeat(x, n_targets, axis=1) if x.shape[1] == 1 else x

    # Compute per-target sums and cross terms
    sx_arr = x.sum(axis=0, dtype=np.float64)
    sx2_arr = (x * x).sum(axis=0, dtype=np.float64)
    sy = target.sum(axis=0, dtype=np.float64)
    sy2 = (target * target).sum(axis=0, dtype=np.float64)
    sxy = (x * target).sum(axis=0, dtype=np.float64)

    if n_samples <= 1:
        raise ValueError('Insufficient samples (<=1) to compute SSE in _calculate_feature_statistics')

    sses = np.zeros(n_targets, dtype=np.float64)
    for i in range(n_targets):
        sses[i] = lin_regression_1variable_with_sums(
            float(sx_arr[i]), float(sy[i]), float(sx2_arr[i]), float(sxy[i]), n_samples, float(sy2[i])
        )

    return FeatureTargetStats(sx=sx_arr, sx2=sx2_arr, sxy=sxy, sses=sses)


def single_feature_score(feature_stats: FeatureTargetStats, baseline_stats: TargetStats) -> float:
    """Score a single feature via SSE reduction: stdev_baseline - stdev_with_feature.

    Args:
        feature_stats: Precomputed sums/SSEs for the feature against target(s)
        baseline_stats: Baseline statistics from calculate_baseline_statistics

    Returns:
        Average improvement score across all targets
    """
    # Check if feature has invalid statistics
    if np.any(np.isinf(feature_stats.sx)):
        raise ValueError('an infinite value was passed')

    # Calculate raw scores for each target: stdev_baseline - stdev_with_feature
    raw_scores = []
    for i in range(len(baseline_stats.stdevs)):
        if np.isinf(feature_stats.sses[i]):
            raise ValueError('an infinite value was passed')
        baseline_stdev = baseline_stats.stdevs[i]
        # Add numerical stability check for SSE
        n = baseline_stats.n_samples
        sse_normalized = feature_stats.sses[i] / n
        if sse_normalized < 0:
            # Numerical precision issue - clamp to zero
            sse_normalized = 0.0
        feature_stdev = np.sqrt(sse_normalized)
        raw_scores.append(baseline_stdev - feature_stdev)

    # Average across all targets, handling infinite values
    finite_scores = [score for score in raw_scores if np.isfinite(score)]
    if len(finite_scores) != len(raw_scores):
        raise ValueError('an infinite value was passed')

    average_score = np.mean(finite_scores)

    return float(average_score)


# ---------------------------------------------------------------------------
# Target and feature preparation functions
# ---------------------------------------------------------------------------

def prepare_targets(target_values: np.ndarray) -> np.ndarray:
    """Convert targets for regression or classification (binary/multiclass),
    using a stable, sorted class ordering via np.unique.

    Returns:
        - Regression/binary: (n_samples, 1)
        - Multiclass: (n_samples, n_classes) one-vs-rest
    """
    # Check if target is numeric (regression) or categorical (classification)
    if target_values.dtype.kind in 'fc':  # float or complex
        # Regression: Ensure 2D shape for downstream code
        return target_values.reshape(-1, 1)
    else:
        # Classification: use np.unique for stable, sorted class ordering
        classes, inv = np.unique(target_values, return_inverse=True)
        n_classes = classes.shape[0]
        if n_classes == 2:  # noqa: PLR2004
            return inv.reshape(-1, 1)
        else:
            # Multi-class: one-vs-rest encoding
            one_hot = np.zeros((len(inv), n_classes))
            one_hot[np.arange(len(inv)), inv] = 1
            return one_hot


def encode_categorical_loo(series: pl.Series, target: np.ndarray) -> np.ndarray:
    """Generalized Leave-One-Out (LOO) target encoding for regression and classification.

    - Regression: Treats the target as a single-class problem.
    - Classification: Handles binary and multiclass targets.

    Returns:
        np.ndarray: An array with shape (n_samples, n_classes).
                    For regression and binary classification, the shape will be (n_samples, 1).
                    The result can be squeezed to (n_samples,) if needed post-call.
    """
    # 1. Standardize Inputs
    if target.ndim == 1:
        target = target.reshape(-1, 1)  # Ensure target is always 2D

    n_samples, n_classes = target.shape

    # Use robust preprocessing for the categorical feature
    series_str = series.cast(pl.Utf8)
    values = series_str.to_numpy()

    # 2. Pre-compute Statistics
    # Global mean for fallback (more robust for singletons)
    global_mean = target.mean(axis=0)

    # Per-category sums and counts
    category_sum: dict[str, np.ndarray] = {}
    category_count: dict[str, int] = {}

    for cat in np.unique(values):
        mask = (values == cat)
        # This works for both regression (n_classes=1) and classification
        category_sum[cat] = target[mask].sum(axis=0)
        category_count[cat] = int(mask.sum())

    # 3. Perform LOO Encoding
    encoded = np.zeros_like(target, dtype=np.float64)

    for i, cat in enumerate(values):
        cat_total_sum = category_sum[cat]
        cat_total_count = category_count[cat]
        current_target_value = target[i]

        if cat_total_count > 1:
            # The core LOO logic works for both scalars and vectors
            encoded[i] = (cat_total_sum - current_target_value) / (cat_total_count - 1)
        else:
            # Use the global mean as a safe fallback for singleton categories
            encoded[i] = global_mean

    # 4. Finalize Output Shape
    # If the original task was regression or binary, squeeze the output to be a 1D array.
    if n_classes == 1:
        return encoded.ravel()  # Shape (n_samples,)

    return encoded  # Shape (n_samples, n_classes) for multiclass


def prepare_feature_values(features: Sequence[Feature], df: pl.DataFrame, target: np.ndarray) -> dict[str, np.ndarray]:
    """Prepare feature values for selection.

    Returns:
        Dict mapping feature names to a single numeric component array per feature.
    """
    feature_values = {}
    for feature in features:
        if feature.name not in df.columns:
            raise ValueError(f'Feature {feature.name} not found in dataset')

        if isinstance(feature, CategoricalFeature):
            # Categorical: direct LOO encoding
            series = df[feature.name]
            encoded = encode_categorical_loo(series, target)
            feature_values[feature.name] = encoded
        else:
            # Numeric: single component
            try:
                numeric_data = df[feature.name].cast(pl.Float64).to_numpy()
                feature_values[feature.name] = numeric_data
            except (pl.exceptions.ComputeError, ValueError):
                # Skip features that cannot be cast to numeric
                continue

    return feature_values


def prepare_sse_data(features: Sequence[Feature], df: pl.DataFrame, target_column: str) -> tuple[np.ndarray, dict[str, np.ndarray], TargetStats]:
    """Prepare target and feature data for SSE calculations.
    
    Args:
        features: List of features to prepare
        df: DataFrame containing feature and target data
        target_column: Name of the target column
        
    Returns:
        Tuple of (prepared_target, feature_values, baseline_stats)
    """
    # Extract and prepare target
    y_raw = df[target_column].to_numpy()
    prepared_target = prepare_targets(y_raw)
    baseline_stats = calculate_baseline_statistics(prepared_target)
    
    # Prepare feature values
    feature_values = prepare_feature_values(features, df, prepared_target)
    
    return prepared_target, feature_values, baseline_stats


# ---------------------------------------------------------------------------
# High-level SSE reduction calculation
# ---------------------------------------------------------------------------

def calculate_sse_reduction(feature: Feature, series: pl.Series, target: pl.Series) -> float:
    """Calculate SSE reduction for a feature against target using linear regression.
    
    Assumes target is already properly formatted (binned for numeric targets, categorical for string targets).
    Handles numeric, boolean, and categorical features with appropriate encoding.
    
    Args:
        feature: Feature object for type information
        series: Feature values as polars Series
        target: Target values as polars Series (already properly formatted)
        
    Returns:
        SSE reduction score (baseline_stdev - feature_stdev)
    """
    # Create aligned arrays without nulls
    df = pl.DataFrame({feature.name: series, 'target': target}).drop_nulls()
    
    # Use shared preparation function
    prepared_target, feature_values, baseline_stats = prepare_sse_data([feature], df, 'target')
    feature_np = feature_values[feature.name]
    
    # Calculate feature statistics
    feature_stats = calculate_feature_statistics(feature_np, prepared_target)
    
    # Calculate SSE reduction score
    return single_feature_score(feature_stats, baseline_stats)
