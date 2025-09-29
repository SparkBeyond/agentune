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
    
    # Standard histogram representation (like numpy.histogram)
    histogram_counts: tuple[int, ...]      # Counts in each bin
    histogram_bin_edges: tuple[float, ...] # Bin edges (length = len(counts) + 1)


@frozen
class NumericRegressionRelationshipStats(RelationshipStats):
    """Enhanced relationship statistics for numeric feature-target combinations in regression."""
    
    # Correlation measures
    pearson_correlation: float
    spearman_correlation: float
