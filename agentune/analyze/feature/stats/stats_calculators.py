from __future__ import annotations

from typing import cast, override

import numpy as np
import polars as pl
from attrs import field, frozen
from scipy.stats import f_oneway, pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif

from agentune.analyze.feature.base import (
    BoolFeature,
    CategoricalFeature,
    Feature,
    NumericFeature,
)
from agentune.analyze.feature.problem import Problem
from agentune.analyze.feature.stats.base import (
    FeatureStats,
    RelationshipStats,
    SyncFeatureStatsCalculator,
    SyncRelationshipStatsCalculator,
)
from agentune.analyze.feature.stats.stats import (
    BooleanClassificationStats,
    BooleanFeatureStats,
    BooleanRegressionStats,
    CategoricalClassificationStats,
    CategoricalFeatureStats,
    CategoricalRegressionStats,
    NumericClassificationStats,
    NumericFeatureStats,
    NumericRegressionStats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class BooleanStatsCalculator(SyncFeatureStatsCalculator[BoolFeature]):
    """Calculator for boolean feature statistics."""

    @override
    def calculate_from_series(self, feature: BoolFeature, series: pl.Series) -> BooleanFeatureStats:
        """Calculate statistics for a boolean feature from a polars Series.

        Args:
            feature: The feature to calculate statistics for
            series: A polars Series containing the feature data

        Returns:
            Boolean feature statistics

        """
        n_total = series.len()
        n_missing = int(series.null_count())

        # Use Polars methods directly instead of converting to NumPy
        non_null_series = series.filter(series.is_not_null()).cast(pl.Boolean)
        support = int(non_null_series.sum()) if non_null_series.len() > 0 else 0
        coverage = support / (n_total - n_missing) if n_total - n_missing > 0 else np.nan
        return BooleanFeatureStats(
            n_total=n_total, n_missing=n_missing, support=support, coverage=coverage
        )


class NumericStatsCalculator(SyncFeatureStatsCalculator[NumericFeature]):
    """Calculator for numeric feature statistics."""

    @override
    def calculate_from_series(self, feature: NumericFeature, series: pl.Series) -> NumericFeatureStats:
        """Calculate statistics for a numeric feature from a polars Series.

        Args:
            feature: The feature to calculate statistics for
            series: A polars Series containing the feature data

        Returns:
            Numeric feature statistics

        """
        n_total = series.len()
        n_missing = int(series.null_count())
        non_null_series = series.drop_nulls()

        # Calculate statistics using Polars methods where possible
        # TODO: decide correct behavior when input includes NaNs. The current behavior is pretty inconsistent,
        # some metrics will be NaN and others will ignore input NaNs.
        if non_null_series.len() == 0:
            min_val = np.nan
            max_val = np.nan
            mean = np.nan
            std = np.nan
            variance = np.nan
            median = np.nan
            q1 = np.nan
            q3 = np.nan
        else:
            # Use Polars descriptive statistics methods
            min_val = cast(float, non_null_series.min())
            max_val = cast(float, non_null_series.max())
            mean = cast(float, non_null_series.mean())

            # Standard deviation and variance
            std = cast(float, non_null_series.std(ddof=1))
            variance = cast(float, non_null_series.var(ddof=1))

            # For quantiles, use Polars quantile method
            median = cast(float, non_null_series.quantile(0.5))
            q1 = cast(float, non_null_series.quantile(0.25))
            q3 = cast(float, non_null_series.quantile(0.75))
        return NumericFeatureStats(
            n_total=n_total,
            n_missing=n_missing,
            min=min_val,
            max=max_val,
            mean=mean,
            std=std,
            variance=variance,
            median=median,
            q1=q1,
            q3=q3,
        )


class CategoricalStatsCalculator(SyncFeatureStatsCalculator[CategoricalFeature]):
    """Calculator for categorical feature statistics."""

    @override
    def calculate_from_series(
        self, feature: CategoricalFeature, series: pl.Series
    ) -> CategoricalFeatureStats:
        """Compute basic categorical statistics for a Polars Series.

        Args:
            feature: The feature to calculate statistics for
            series: A polars Series containing the feature data

        Returns:
            Categorical feature statistics

        """
        n_total = series.len()
        n_missing = series.null_count()

        # Collect value counts only for the non-null part
        non_null_series = series.drop_nulls()

        # Get value counts as a dictionary mapping values to their counts
        # Empty series will just produce an empty dict
        counts: dict[str, int] = dict(non_null_series.value_counts().rows())
        all_counts = { k: counts.get(k, 0) for k in feature.categories }

        return CategoricalFeatureStats(
            n_total=n_total, n_missing=n_missing, unique_count=len(counts), value_counts=all_counts
        )


@frozen
class CombinedSyncFeatureStatsCalculator(SyncFeatureStatsCalculator[Feature]):
    boolean_stats_calculator: SyncFeatureStatsCalculator[BoolFeature] = field(default=BooleanStatsCalculator())
    numeric_stats_calculator: SyncFeatureStatsCalculator[NumericFeature] = field(default=NumericStatsCalculator())
    categorical_stats_calculator: SyncFeatureStatsCalculator[CategoricalFeature] = field(default=CategoricalStatsCalculator())
    
    @override
    def calculate_from_series(self, feature: Feature, series: pl.Series) -> FeatureStats[Feature]:
        if isinstance(feature, BoolFeature):
            return self.boolean_stats_calculator.calculate_from_series(feature, series)
        elif isinstance(feature, NumericFeature):
            return self.numeric_stats_calculator.calculate_from_series(feature, series)
        elif isinstance(feature, CategoricalFeature):
            return self.categorical_stats_calculator.calculate_from_series(feature, series)
        else:
            raise TypeError(f'Unsupported feature type: {type(feature)}')

class BooleanRegressionCalculator(SyncRelationshipStatsCalculator[BoolFeature]):
    """Calculator for boolean feature regression statistics."""

    @override
    def calculate_from_series(
        self, feature: BoolFeature, series: pl.Series, target: pl.Series, problem: Problem
    ) -> BooleanRegressionStats:
        """Calculate regression statistics for a boolean feature from polars Series.

        Args:
            feature: The feature to calculate statistics for
            series: A polars Series containing the feature data
            target: A polars Series containing the target data

        Returns:
            Boolean regression statistics

        """
        # Filter out nulls and prepare data
        n_missing = int(series.null_count())
        mask = series.is_not_null()
        valid_series = series.filter(mask)
        valid_target = target.filter(mask).cast(pl.Float64)
        n_samples = valid_series.len()

        # For the Pearson correlation, we need numpy
        x = valid_series.to_numpy()
        y = valid_target.to_numpy()

        # Calculate stats using Polars where possible
        true_mask = valid_series
        false_mask = ~valid_series
        n_true = int(valid_series.sum())
        n_false = n_samples - n_true

        # Calculate means with proper type handling
        true_mean = cast(float, valid_target.filter(true_mask).mean()) if n_true > 0 else np.nan
        false_mean = cast(float, valid_target.filter(false_mask).mean()) if n_false > 0 else np.nan
        overall_mean = cast(float, valid_target.mean())

        # Calculate shift with proper type handling
        if n_true == n_samples:
            shift = 0.0
        elif n_true == 0:
            # TODO: I disagree. Every user of this code will need to treat this as identical to 0.0.
            shift = np.nan
        else:
            shift = float(true_mean - overall_mean if not np.isnan(true_mean) else np.nan)

        # Pearson correlation still requires NumPy as Polars doesn't have this
        corr, _ = pearsonr(x, y) if n_samples > 1 else (np.nan, np.nan)
        return BooleanRegressionStats(
            n_samples=n_samples,
            n_missing_feature=n_missing,
            mean_true=true_mean,
            mean_false=false_mean,
            shift=shift,
            point_biserial=float(corr),
        )


class BooleanClassificationCalculator(SyncRelationshipStatsCalculator[BoolFeature]):
    """Calculator for boolean feature classification statistics."""
    
    @override
    def calculate_from_series(
        self, feature: BoolFeature, series: pl.Series, target: pl.Series, problem: Problem
    ) -> BooleanClassificationStats:
        """Calculate classification statistics for a boolean feature from polars Series.

        Args:
            feature: The feature to calculate statistics for
            series: A polars Series containing the feature data
            target: A polars Series containing the target data

        Returns:
            Boolean classification statistics

        """
        # Filter out nulls from both series
        mask = series.is_not_null() & target.is_not_null()
        n_samples = int(mask.sum())
        n_missing = int(series.null_count())

        valid_series = series.filter(mask)
        valid_target = target.filter(mask)

        # Get unique classes using Polars
        # TODO: Classes should be ordered by a global definition, not sorted. Pending global class order definition.
        classes = valid_target.unique().sort().to_list()

        # For mutual information we need numpy
        f_np = valid_series.to_numpy()
        y_np = valid_target.to_numpy()

        # Calculate class statistics - we need numpy for these computations
        # since we need to group by target values
        true_counts = []
        class_counts = []

        for cls in classes:
            # Count occurrences of each class
            cls_mask = valid_target == cls
            cls_count = int(cls_mask.sum())
            class_counts.append(cls_count)

            # Count True values for this class
            true_count = int(valid_series.filter(cls_mask).sum())
            true_counts.append(true_count)

        # Convert to arrays for calculations
        true_counts_array = np.array(true_counts)
        class_counts_array = np.array(class_counts)

        # Percent of samples in each class that is True for feature
        coverage = true_counts_array / class_counts_array

        # Info gain (mutual information) - requires numpy
        mi = mutual_info_classif(f_np.reshape(-1, 1), y_np, discrete_features=True)[0]

        # Shift and lift calculations
        overall_true_rate = cast(float, valid_series.mean())

        # Ensure coverage is a proper numpy array of floats
        shift = (
            coverage - overall_true_rate
            if overall_true_rate > 0
            else np.array([np.nan] * len(classes))
        )

        # Lift values with proper type handling
        lift_values = (
            coverage / overall_true_rate
            if overall_true_rate > 0
            else np.array([np.nan] * len(classes))
        )
        lift = lift_values.tolist()
        return BooleanClassificationStats(
            n_samples=n_samples,
            n_missing_feature=n_missing,
            true_counts=tuple(true_counts),
            categories=tuple(classes),
            info_gain=float(mi),
            lift=tuple(lift),
            shift=tuple(shift.tolist()),
        )


class NumericRegressionCalculator(SyncRelationshipStatsCalculator[NumericFeature]):
    """Calculator for numeric feature regression statistics."""

    @override
    def calculate_from_series(
        self, feature: NumericFeature, series: pl.Series, target: pl.Series, problem: Problem
    ) -> NumericRegressionStats:
        """Calculate regression statistics for a numeric feature from polars Series.

        Args:
            feature: The feature to calculate statistics for
            series: A polars Series containing the feature data
            target: A polars Series containing the target data

        Returns:
            Numeric regression statistics

        """
        # Filter out nulls and prepare data
        mask = series.is_not_null() & target.is_not_null()
        valid_series = series.filter(mask).cast(pl.Float64)
        valid_target = target.filter(mask).cast(pl.Float64)
        n_samples = valid_series.len()
        n_missing = int(series.null_count())

        # For correlation calculations, we need numpy
        # as Polars doesn't have these statistical functions
        if n_samples > 1:
            # Convert to numpy only for the correlation calculations
            x = valid_series.to_numpy()
            y = valid_target.to_numpy()
            p_corr, p_val = pearsonr(x, y)
            s_corr, s_val = spearmanr(x, y) if n_samples > 1 else (np.nan, np.nan)
        else:
            p_corr, p_val = np.nan, np.nan
            s_corr, s_val = np.nan, np.nan

        return NumericRegressionStats(
            n_samples=n_samples,
            n_missing_feature=n_missing,
            pearson_r=float(p_corr),
            pearson_p=float(p_val),
            spearman_r=float(s_corr),
            spearman_p=float(s_val),
        )


class NumericClassificationCalculator(
    SyncRelationshipStatsCalculator[NumericFeature]
):
    """Calculator for numeric feature classification statistics."""

    @override
    def calculate_from_series(
        self, feature: NumericFeature, series: pl.Series, target: pl.Series, problem: Problem
    ) -> NumericClassificationStats:
        """Calculate classification statistics for a numeric feature from polars Series.

        Args:
            feature: The feature to calculate statistics for
            series: A polars Series containing the feature data
            target: A polars Series containing the target data

        Returns:
            Numeric classification statistics

        """
        # Filter out nulls and prepare data
        mask = series.is_not_null() & target.is_not_null()
        valid_series = series.filter(mask).cast(pl.Float64)
        valid_target = target.filter(mask)
        n_samples = valid_series.len()
        n_missing = int(series.null_count())

        # For ANOVA calculations, we need numpy as Polars doesn't have this statistical function
        # Convert to numpy only for the calculations that require it
        x = valid_series.to_numpy()
        y = valid_target.to_numpy()

        # Get unique classes and group data by class
        unique_classes = valid_target.unique().to_list()
        groups = [x[y == cls] for cls in unique_classes]

        # Calculate ANOVA F-statistic
        f_stat, p = f_oneway(*groups) if len(groups) > 1 else (np.nan, np.nan)

        return NumericClassificationStats(
            n_samples=n_samples,
            n_missing_feature=n_missing,
            anova_f=float(f_stat),
            p_value=float(p),
        )


class CategoricalRegressionCalculator(
    SyncRelationshipStatsCalculator[CategoricalFeature]
):
    """Calculator for categorical feature regression statistics."""

    @override
    def calculate_from_series(
        self, feature: CategoricalFeature, series: pl.Series, target: pl.Series, problem: Problem
    ) -> CategoricalRegressionStats:
        """Calculate regression stats for a categorical feature.
        Assumptions:
        - No previously unseen categories in the feature. For now - raise ValueError if any are found.
          TODO: decide whether a categorical feature's categories list should always include a special "unknown"
                value; all unexpected values returned by the feature would be converted to this value, 
                and the API and categorical stats calculation should be adjusted accordingly.
                Or, whether a feature returning an unexpected value should be an error and replaced by a missing value.
                (Either way this should happen in the CategoricalFeature implementation and not here.)
        - Categories not present in data are included in the result with 0 count and mean.
        """
        feature_categories = tuple(feature.categories)

        # Keep rows where *both* feature and target are non-null
        df = pl.DataFrame({'cat': series, 'target': target}).drop_nulls()

        n_samples: int = df.height
        n_missing: int = int(series.null_count())

        if n_samples == 0:
            # Degenerate edge-case → return NaNs/empties
            return CategoricalRegressionStats(
                n_samples=0,
                n_missing_feature=n_missing,
                category_counts=(),
                categories=feature_categories,
                mean_by_category=(),
                shift_by_category=(),
            )

        # One groupby does everything: count & mean of target per category
        agg_df = (
            df.lazy()
            .drop_nulls()
            .group_by('cat')
            .agg([pl.len().alias('count'), pl.mean('target').alias('mean')])
            .sort('cat')
            .collect()
        )

        cats = agg_df['cat'].to_list()
        counts = [int(c) for c in agg_df['count'].to_list()]
        means = [float(m) for m in agg_df['mean'].to_list()]

        # Raise error if any categories are missing
        missing_cats = set(feature_categories) - set(cats)
        if missing_cats:
            raise ValueError(f'Missing categories: {missing_cats}')

        # Sort categories to match feature categories order, and fill missing categories with 0 count and mean
        sorted_cat_counts_means = []
        for cat in feature_categories:
            if cat in cats:
                sorted_cat_counts_means.append(
                    (cat, counts[cats.index(cat)], means[cats.index(cat)])
                )
            else:
                sorted_cat_counts_means.append((cat, 0, float('nan')))

        # Calculate overall mean
        overall_mean = cast(float, df['target'].mean())
        shifts = [m - overall_mean for m in means]

        return CategoricalRegressionStats(
            n_samples=n_samples,
            n_missing_feature=n_missing,
            category_counts=tuple(counts),
            categories=tuple(cats),
            mean_by_category=tuple(means),
            shift_by_category=tuple(shifts),
        )


class CategoricalClassificationCalculator(
    SyncRelationshipStatsCalculator[CategoricalFeature]
):
    """Calculator for categorical feature classification statistics."""

    @override
    def calculate_from_series(
        self, feature: CategoricalFeature, series: pl.Series, target: pl.Series, problem: Problem
    ) -> CategoricalClassificationStats:
        """Classification stats for a categorical feature."""
        df = pl.DataFrame({'cat': series, 'class': target}).drop_nulls()

        n_missing: int = int(series.null_count())
        n_samples: int = df.height

        if n_samples == 0:
            return CategoricalClassificationStats(
                n_samples=0,
                n_missing_feature=n_missing,
                categories=tuple(feature.categories),
                classes=(),
                category_class_counts=(),
                info_gain=np.nan,
                lift=(),
                shift=(),
            )

        # Get the unique categories and classes while preserving their types
        cats = tuple(feature.categories)
        unique_classes = df['class'].unique().sort().to_list()

        # Count occurrences for each category-class combination
        counts = df.group_by(['cat', 'class']).len().sort(['cat', 'class'])

        # Initialize the contingency matrix with zeros
        matrix = np.zeros((len(cats), len(unique_classes)), dtype=int)

        # Fill in the matrix using the counts
        # TODO: implement more efficiently, avoiding loops
        for row in counts.iter_rows(named=True):
            cat_idx = cats.index(row['cat'])
            class_idx = unique_classes.index(row['class'])
            matrix[cat_idx, class_idx] = row['len']

        # For consistency with variable names in the rest of the function
        classes = unique_classes

        # mutual information needs NumPy arrays
        mi = mutual_info_classif(
            df['cat'].to_numpy().reshape(-1, 1), df['class'].to_numpy(), discrete_features=True
        )[0]

        # lift / shift
        p_cat = matrix.sum(axis=1) / n_samples  # (k,)
        p_class = matrix.sum(axis=0) / n_samples  # (c,)
        p_cat_class = matrix / n_samples  # (k, c)

        lift = (p_cat_class / np.outer(p_cat, p_class)).tolist()
        shift = (p_cat_class - p_class).tolist()

        return CategoricalClassificationStats(
            n_samples=n_samples,
            n_missing_feature=n_missing,
            categories=cats,
            classes=tuple(classes),
            category_class_counts=tuple(tuple(int(x) for x in row) for row in matrix.tolist()),
            info_gain=float(mi),
            lift=tuple(tuple(float(x) for x in row) for row in lift),
            shift=tuple(tuple(float(x) for x in row) for row in shift),
        )

@frozen
class CombinedSyncRelationshipStatsCalculator(SyncRelationshipStatsCalculator[Feature]):
    boolean_calculator: SyncRelationshipStatsCalculator[BoolFeature]
    numeric_calculator: SyncRelationshipStatsCalculator[NumericFeature]
    categorical_calculator: SyncRelationshipStatsCalculator[CategoricalFeature]
    
    @override
    def calculate_from_series(self, feature: Feature, series: pl.Series, target: pl.Series,
                              problem: Problem) -> RelationshipStats[Feature]:
        if isinstance(feature, BoolFeature):
            return self.boolean_calculator.calculate_from_series(feature, series, target, problem)
        elif isinstance(feature, NumericFeature):
            return self.numeric_calculator.calculate_from_series(feature, series, target, problem)
        elif isinstance(feature, CategoricalFeature):
            return self.categorical_calculator.calculate_from_series(feature, series, target, problem)
        else:
            raise TypeError(f'Unsupported feature type: {type(feature)}')
        
default_regression_calculator = CombinedSyncRelationshipStatsCalculator(
    boolean_calculator=BooleanRegressionCalculator(),
    numeric_calculator=NumericRegressionCalculator(),
    categorical_calculator=CategoricalRegressionCalculator(),
)

default_classification_calculator = CombinedSyncRelationshipStatsCalculator(
    boolean_calculator=BooleanClassificationCalculator(),
    numeric_calculator=NumericClassificationCalculator(),
    categorical_calculator=CategoricalClassificationCalculator(),
)

default_feature_stats_calculator = CombinedSyncFeatureStatsCalculator()
