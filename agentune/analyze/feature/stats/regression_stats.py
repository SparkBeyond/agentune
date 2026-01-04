"""Regression statistics for numeric features and targets.

This module provides enhanced statistics data classes for numeric feature-target combinations
in regression problems, including histograms and correlation measures.

"""

from attrs import frozen

from agentune.analyze.feature.stats.base import (
    FeatureStats,
    RelationshipStats,
)


@frozen
class NumericFeatureStats(FeatureStats):
    """Enhanced feature statistics for numeric features in regression problems."""
    
    # Infinity and finite counts (specific to numeric data)
    n_finite: int  # Number of finite values
    n_positive_infinite: int  # Number of positive infinite values
    n_negative_infinite: int  # Number of negative infinite values
    n_nan: int  # Number of NaN values (distinct from missing/null)
    
    # Standard histogram representation (like numpy.histogram)
    histogram_counts: tuple[int, ...]      # Counts in each bin
    histogram_bin_edges: tuple[float, ...] # Bin edges (length = len(counts) + 1)


@frozen
class NumericRegressionRelationshipStats(RelationshipStats):
    """Enhanced relationship statistics for numeric feature-target combinations in regression."""
    
    # Correlation measures
    pearson_correlation: float
    spearman_correlation: float
