"""Joint utility for single feature SSE reduction calculations.

This module provides shared functionality for calculating Sum of Squared Errors (SSE)
reduction when computing individual features against targets. It supports both
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
    
    When the system is singular (feature has no variance), returns (0, b2/a22)
    which corresponds to predicting the target mean.
    """
    matrix_a = np.array([[a11, a12], [a21, a22]])
    b = np.array([b1, b2])
    try:
        solution = np.linalg.solve(matrix_a, b)
        return tuple(solution)
    except np.linalg.LinAlgError:
        # Singular system - feature has no variance
        # Return mean-only solution: a=0, c=mean_target
        # From the second equation: sx*a + n*c = sy, with a=0: c = sy/n = b2/a22
        if a22 != 0:
            return (0.0, b2 / a22)
        else:
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


def single_feature_score(feature_stats: FeatureTargetStats, baseline_stats: TargetStats) -> tuple[float, float]:
    """Score a single feature via SSE reduction and R².

    Args:
        feature_stats: Precomputed sums/SSEs for the feature against target(s)
        baseline_stats: Baseline statistics from calculate_baseline_statistics

    Returns:
        Tuple of (sse_reduction, r_squared):
        - sse_reduction: baseline_stdev - feature_stdev (average across targets)
        - r_squared: 1 - (SSE_feature / SSE_baseline), clamped to range [0, 1] (average across targets).
          Negative R² values (feature performs worse than baseline) are clamped to 0.
    """
    # Check if feature has invalid statistics
    if np.any(np.isinf(feature_stats.sx)):
        raise ValueError('an infinite value was passed')

    # Calculate both metrics for each target in a single pass
    sse_reductions = []
    r2_scores = []
    
    for i in range(len(baseline_stats.stdevs)):
        if np.isinf(feature_stats.sses[i]):
            raise ValueError('an infinite value was passed')
        
        baseline_sse = baseline_stats.priorsses[i]
        feature_sse = feature_stats.sses[i]
        n = baseline_stats.n_samples
        
        # Calculate SSE reduction (stdev_baseline - stdev_feature)
        baseline_stdev = baseline_stats.stdevs[i]
        sse_normalized = max(0.0, feature_sse / n)  # Clamp for numerical stability
        feature_stdev = np.sqrt(sse_normalized)
        sse_reductions.append(baseline_stdev - feature_stdev)
        
        # Calculate R² = 1 - (SSE_feature / SSE_baseline)
        if baseline_sse == 0:
            r2_scores.append(0.0)
        else:
            r2 = 1.0 - (feature_sse / baseline_sse)
            r2_scores.append(max(0.0, min(1.0, r2)))  # Clamp to [0, 1]

    # Average across all targets, handling infinite values
    finite_sse = [s for s in sse_reductions if np.isfinite(s)]
    finite_r2 = [s for s in r2_scores if np.isfinite(s)]
    
    if len(finite_sse) != len(sse_reductions) or len(finite_r2) != len(r2_scores):
        raise ValueError('an infinite value was passed')

    avg_sse_reduction = float(np.mean(finite_sse))
    avg_r2 = float(np.mean(finite_r2))

    return avg_sse_reduction, avg_r2


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


def encode_categorical_catboost(series: pl.Series, target: np.ndarray, n_permutations: int = 5, prior: float = 5.0, random_seed: int = 42) -> np.ndarray:
    """CatBoost-style Ordered Target Statistics encoding for categorical features.
    
    This implements the CatBoost algorithm which avoids target leakage by:
    1. Creating random permutations of the data
    2. For each sample, computing target statistics using only samples that appear
       BEFORE it in the permutation order (ordered target statistics)
    3. Averaging results across multiple permutations to reduce variance
    
    The encoding uses prior smoothing: (cumsum + prior*global_mean) / (count + prior)
    This acts as if we've observed 'prior' additional pseudo-samples with the global mean value.
    
    Args:
        series: Categorical feature values as polars Series
        target: Target values as numpy array (1D or 2D)
        n_permutations: Number of random permutations to average (default: 5)
        prior: Prior value for smoothing (default: 5.0)
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        np.ndarray: Encoded values with shape (n_samples, n_classes) or (n_samples,) for single-class
    """
    # 1. Standardize Inputs
    if target.ndim == 1:
        target = target.reshape(-1, 1)  # Ensure target is always 2D

    n_samples, n_classes = target.shape

    # Use robust preprocessing for the categorical feature
    series_str = series.cast(pl.Utf8)
    values = series_str.to_numpy()
    
    # Global mean as prior
    global_mean = target.mean(axis=0)
    
    # Get unique categories for encoding
    unique_cats = np.unique(values)
    
    # Perform CatBoost Ordered Target Statistics
    # Initialize result array to accumulate encodings from multiple permutations
    result = np.zeros((n_samples, n_classes), dtype=np.float64)
    
    rng = np.random.RandomState(random_seed)
    
    for _ in range(n_permutations):
        # Create random permutation
        perm = rng.permutation(n_samples)
        
        # Track cumulative statistics for each category as we iterate
        # category -> (sum_of_targets, count)
        cumulative_sum: dict[str, np.ndarray] = {}
        cumulative_count: dict[str, int] = {}
        
        # Initialize all categories with prior
        for cat in unique_cats:
            cumulative_sum[cat] = global_mean * prior
            cumulative_count[cat] = 0
        
        # Iterate through samples in permuted order
        for idx in perm:
            cat = values[idx]
            
            # Encode current sample using statistics from samples seen so far
            # Formula: cumsum / (count + prior), where cumsum is initialized with prior*global_mean
            count = cumulative_count[cat]
            result[idx] += cumulative_sum[cat] / (count + prior)
            # Update cumulative statistics with current sample
            cumulative_sum[cat] += target[idx]
            cumulative_count[cat] += 1
    
    # 3. Average across permutations
    result /= n_permutations
    
    # 4. Finalize Output Shape
    if n_classes == 1:
        return result.ravel()  # Shape (n_samples,)
    
    return result  # Shape (n_samples, n_classes) for multiclass


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
            encoded = encode_categorical_catboost(series, target)
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

def calculate_sse_reduction(feature: Feature, series: pl.Series, target: pl.Series) -> tuple[float, float]:
    """Calculate both SSE reduction and R² for a feature against target using linear regression.
    
    Assumes target is already properly formatted (binned for numeric targets, categorical for string targets).
    Handles numeric, boolean, and categorical features with appropriate encoding.
    
    Args:
        feature: Feature object for type information
        series: Feature values as polars Series
        target: Target values as polars Series (already properly formatted)
        
    Returns:
        Tuple of (sse_reduction, r_squared):
        - sse_reduction: baseline_stdev - feature_stdev
        - r_squared: 1 - (SSE_feature / SSE_baseline), clamped to be in range [0, 1], nan is mapped to 0.
    """
    # Create aligned arrays without nulls
    df = pl.DataFrame({feature.name: series, 'target': target}).drop_nulls()
    
    # Check if we have enough samples after dropping nulls
    if len(df) <= 1:
        raise ValueError(f'Insufficient samples for SSE calculation: {len(df)} sample(s) after removing nulls. Need at least 2 samples.')
    
    # Use shared preparation function
    prepared_target, feature_values, baseline_stats = prepare_sse_data([feature], df, 'target')
    feature_np = feature_values[feature.name]
    
    # Calculate feature statistics and both metrics in one pass
    feature_stats = calculate_feature_statistics(feature_np, prepared_target)
    return single_feature_score(feature_stats, baseline_stats)
