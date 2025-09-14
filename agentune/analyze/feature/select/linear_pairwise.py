 
import logging
from collections.abc import Sequence

import numpy as np
import polars as pl
from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.core.dataset import DatasetSource
from agentune.analyze.feature.base import CategoricalFeature, Feature
from agentune.analyze.feature.problem import Problem, TargetKind
from agentune.analyze.feature.select.base import SyncEnrichedFeatureSelector
from agentune.analyze.feature.util import substitute_default_values

logger = logging.getLogger(__name__)


@frozen
class TargetStats:
    sy: np.ndarray
    sy2: np.ndarray
    priorsses: np.ndarray
    stdevs: np.ndarray
    n_samples: float


@frozen
class FeatureTargetStats:
    sx: np.ndarray
    sx2: np.ndarray
    sxy: np.ndarray
    sses: np.ndarray


_EPSILON = 1e-10

class LinearPairWiseFeatureSelector(SyncEnrichedFeatureSelector):
    """Feature selector using SSE-reduction single and pairwise scoring with closed-form linear regression (no weights).

    Pairwise marginalization updates each candidate's marginal only against the newly selected feature at each
    iteration (incremental update).

    Categorical handling:
    - Leave-one-out target encoding (regression: mean; classification: class-rate similarity or label mean for binary).
    - Single continuous feature scored via SSE reduction.
    - No one-hot/dummy expansion: each categorical feature is encoded into a single numeric column; selection and
      importances are reported at the original feature level.
    """
    def __init__(
        self,
        top_k: int = 10,
        min_marginal_reduction_threshold: float = 1e-8
    ):
        """Initialize selector.

        Args:
            top_k: Number of features to select.
            task_type: 'classification' or 'regression'.
            min_marginal_reduction_threshold: Minimum marginal error reduction required to add a feature.
                Features with lower marginal improvement will not be selected. 
        """
        self.top_k = top_k
        self.min_marginal_reduction_threshold = min_marginal_reduction_threshold

        # No internal model; selector is model-agnostic
        self.final_importances_: dict[str, list] | None = None
        self._selected_feature_names: list[str] | None = None

    def select_features(
        self,
        features: Sequence[Feature],
        enriched_data: DatasetSource,
        problem: Problem,
        conn: DuckDBPyConnection,
    ) -> Sequence[Feature]:
        """Select features using the enriched API, matching the base interface signature.

        Args:
            features: Sequence of Feature definitions.
            enriched_data: DatasetSource containing materialized feature columns and target.
            target_column: Name of the target column in the enriched dataset.
            conn: DuckDB connection/cursor used to access the enriched dataset.
        """
        dataset = enriched_data.to_dataset(conn)
        imputed_dataset = substitute_default_values(dataset, features)
        df_pl = imputed_dataset.data

        target_column = problem.target_column.name
        if target_column not in df_pl.columns:
            raise ValueError(f'Target column {target_column} not found in dataset')
        # If no features, return empty selection
        if not features:
            self.final_importances_ = {'feature': [], 'importance': []}
            return []
        # Make sure features and target align in size:
        if df_pl.shape != enriched_data.to_dataset(conn).data.shape:
            raise ValueError('Feature and target size mismatch')

        y_raw = df_pl[target_column].to_numpy()
        target = self._prepare_targets(y_raw, problem.target_kind)
        baseline_stats = self._calculate_baseline_statistics(target)

        feature_values = self._prepare_feature_values(features, df_pl, target)

        remaining_feature_names = list(feature_values.keys())

        if not remaining_feature_names:
            self.final_importances_ = {'feature': [], 'importance': []}
            return []

        selected_feature_names, feature_scores = self._select_best_features(feature_values, target, baseline_stats)
        selected_features = [f for f in features if f.name in set(selected_feature_names)]

        # Sort by using lambda to ensure stable sort by name for equal importances
        sorted_importances = sorted(selected_feature_names, key=lambda name: (feature_scores[name], name), reverse=True)

        self.final_importances_ = {
            'feature': sorted_importances,
            'importance': [feature_scores[name] for name in sorted_importances]
        }

        return selected_features

    def _prepare_targets(self, target_values: np.ndarray, target_kind: TargetKind) -> np.ndarray:
        """Convert targets for regression or classification (binary/multiclass),
        using a stable, sorted class ordering via np.unique.

        Returns:
            - Regression/binary: (n_samples, 1)
            - Multiclass: (n_samples, n_classes) one-vs-rest
        """
        if target_kind == 'regression':
            # Ensure 2D shape for downstream code
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

    @staticmethod
    def _single_feature_score(feature_stats: FeatureTargetStats, baseline_stats: TargetStats) -> float:
        """Score a single feature via SSE reduction: stdev_baseline - stdev_with_feature.

        Args:
            feature_stats: Precomputed sums/SSEs for the feature against target(s)
            baseline_stats: Baseline statistics from _calculate_baseline_statistics

        Returns:
            Average normalized improvement score across all targets (0-1 range, interpretable as fraction of variance explained)
        """
        # Check if feature has invalid statistics
        if np.any(np.isinf(feature_stats.sx)):
            raise ValueError('an infinite value was passed')

        # Calculate raw scores for each target: stdev_baseline - stdev_with_feature
        raw_scores = []
        overall_baseline_stdev = np.mean(baseline_stats.stdevs)
        if overall_baseline_stdev <= _EPSILON:
            return 0.0
            
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
            
            raw_improvement = baseline_stdev - feature_stdev
            normalized_score = raw_improvement / overall_baseline_stdev

            raw_scores.append(normalized_score)

        finite_scores = [score for score in raw_scores if np.isfinite(score)]
        if len(finite_scores) != len(raw_scores):
            raise ValueError('an infinite value was passed')

        # Calculate average score
        average_score = np.mean(finite_scores)

        return float(average_score)

    @staticmethod
    def _pairwise_feature_score(candidate_stats: FeatureTargetStats, selected_stats: FeatureTargetStats,
                                baseline_stats: TargetStats, sxz: np.ndarray) -> float:
        """Score feature pair via closed-form two-variable regression.
        Computes mean over targets of normalized improvement: (stdev(selected-alone) - stdev(joint)) / stdev_baseline.
        Unweighted sums; stdev = sqrt(SSE / n). Normalized by baseline standard deviation for scale invariance.
        """
        # Check if either feature has invalid statistics
        if (np.any(np.isinf(candidate_stats.sx)) or np.any(np.isinf(selected_stats.sx))):
            raise ValueError('an infinite value was passed')

        n_targets = len(baseline_stats.sy)
        n = baseline_stats.n_samples

        # Calculate joint SSEs using 2-variable linear regression
        joint_sses = []
        for i in range(n_targets):
            # Use the 2-variable linear regression function with sums
            sx = float(candidate_stats.sx[i])
            sz = float(selected_stats.sx[i])  # sz is the selected feature's sx
            sy = float(baseline_stats.sy[i])
            sx2 = float(candidate_stats.sx2[i])
            sz2 = float(selected_stats.sx2[i])
            sxy = float(candidate_stats.sxy[i])
            szy = float(selected_stats.sxy[i])
            sy2 = float(baseline_stats.sy2[i])

            # Call the 2-variable regression function
            joint_sses.append(LinearPairWiseFeatureSelector._lin_regression_2variables_with_sums(
                sx, sz, sy, sx2, sz2, float(sxz[i]), sxy, szy, n, sy2
            ))

        # Calculate improvement scores: stdev_selected_alone - stdev_joint
        unadjusted_scores = []
        overall_baseline_stdev = np.mean(baseline_stats.stdevs)
        if overall_baseline_stdev <= _EPSILON:
            return 0.0
            
        for i in range(n_targets):
            if np.isinf(selected_stats.sses[i]) or np.isinf(joint_sses[i]):
                raise ValueError('an infinite value was passed')

            # Add numerical stability check for SSE
            selected_sse_normalized = selected_stats.sses[i] / n
            joint_sse_normalized = joint_sses[i] / n
            if selected_sse_normalized < 0:
                selected_sse_normalized = 0.0
            if joint_sse_normalized < 0:
                joint_sse_normalized = 0.0
            selected_stdev = np.sqrt(selected_sse_normalized)
            joint_stdev = np.sqrt(joint_sse_normalized)

            normalized_score = (selected_stdev - joint_stdev) / overall_baseline_stdev
            unadjusted_scores.append(normalized_score)

        finite_scores = [score for score in unadjusted_scores if np.isfinite(score)]
        if len(finite_scores) != len(unadjusted_scores):
            raise ValueError('an infinite value was passed')

        average_score = np.mean(finite_scores)

        # Note: Complexity penalty can be added later if needed
        return float(average_score)

    @staticmethod
    def _solve_3x3_system(a11: float, a12: float, a13: float, a21: float, a22: float, a23: float,
                          a31: float, a32: float, a33: float, b1: float, b2: float, b3: float) -> tuple:
        """Solve 3x3 linear system Ax = b using numpy.linalg.solve.
        Returns (x1, x2, x3) or zeros if singular/ill-conditioned.
        """
        matrix_a = np.array([[a11, a12, a13],
                      [a21, a22, a23],
                      [a31, a32, a33]])
        b = np.array([b1, b2, b3])

        try:
            x = np.linalg.solve(matrix_a, b)
            return tuple(x)
        except np.linalg.LinAlgError:
            # Matrix is singular or ill-conditioned
            return (0.0, 0.0, 0.0)

    @staticmethod
    def _solve_2x2_system(a11: float, a12: float, a21: float, a22: float, b1: float, b2: float) -> tuple:
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

    @staticmethod
    def _preprocess_data_array(data: np.ndarray) -> np.ndarray:
        """Validate data arrays contain only finite values; rely on upstream imputation."""
        if not np.all(np.isfinite(data)):
            raise ValueError('an infinite value was passed')
        return data

    def _prepare_feature_values(self, features: Sequence[Feature], df: pl.DataFrame, target: np.ndarray) -> dict[str, np.ndarray]:
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
                encoded = self._encode_categorical_loo(series, target)
                feature_values[feature.name] = encoded
            else:
                # Numeric: single component
                try:
                    numeric_data = df[feature.name].cast(pl.Float64).to_numpy()
                    feature_values[feature.name] = numeric_data
                except (pl.exceptions.ComputeError, ValueError):
                    logger.debug(f'Skipping feature {feature.name} - cannot cast to numeric')
                    continue

        return feature_values

    @staticmethod
    def _create_category_mask(series: pl.Series, cat_value: str | None, exclude_index: int) -> np.ndarray:
        """Create boolean mask for category values, excluding specified index."""
        if cat_value is None:
            cat_mask = series.is_null().to_numpy()
        else:
            cat_mask = (series == cat_value).to_numpy()
        cat_mask[exclude_index] = False
        return cat_mask.astype(bool)

    @staticmethod
    def _encode_categorical_loo(series: pl.Series, target: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def _normalize_to_targets(values: np.ndarray, n_targets: int) -> np.ndarray:
        """Normalize 1D or single-column arrays to (n_samples, n_targets); pass through multi-target arrays."""
        x = values[:, None] if values.ndim == 1 else values
        return np.repeat(x, n_targets, axis=1) if x.shape[1] == 1 else x

    def _select_best_features(self, feature_values: dict[str, np.ndarray], target: np.ndarray, baseline_stats: TargetStats) -> tuple[list[str], dict[str, float]]:
        """Two-phase feature selection: single feature scoring, then pairwise marginalization."""
        remaining_feature_names = list(feature_values.keys())
        selected_feature_names: list[str] = []
        feature_scores: dict[str, float] = {}

        # Precompute per-feature statistics once for reuse in scoring (local per-run cache)
        stats_by_name: dict[str, FeatureTargetStats] = {
            fname: self._calculate_feature_statistics(vals, target) for fname, vals in feature_values.items()
        }

        # Phase 1: single feature scoring and initial filtering
        candidate_marginal_scores: dict[str, float] = {}
        for f_name in remaining_feature_names:
            score = self._single_feature_score(stats_by_name[f_name], baseline_stats)
            candidate_marginal_scores[f_name] = score
        if not remaining_feature_names:
            self.final_importances_ = {'feature': [], 'importance': []}
            return [], {}
        
        # Filter out features that don't meet the threshold (they can never improve)
        viable_features = [
            f_name for f_name in remaining_feature_names 
            if candidate_marginal_scores[f_name] > 0.0 and candidate_marginal_scores[f_name] >= self.min_marginal_reduction_threshold
        ]
        
        if not viable_features:
            # No features meet the threshold, return empty selection
            return [], {}

        # Select best first feature from viable candidates
        best_first_feature = max(viable_features, key=lambda name: (candidate_marginal_scores[name], name))
        best_first_score = candidate_marginal_scores[best_first_feature]
        
        selected_feature_names.append(best_first_feature)
        feature_scores[best_first_feature] = best_first_score
        viable_features.remove(best_first_feature)

        # Phase 2: iteratively add features using pairwise scoring
        # Precompute target-shape constants and per-feature normalization outside the loop
        n_targets = target.shape[1]
        norm_feature_values: dict[str, np.ndarray] = {
            name: self._normalize_to_targets(vals, n_targets) for name, vals in feature_values.items()
        }
        while len(selected_feature_names) < self.top_k and viable_features:
            last_selected_name = selected_feature_names[-1]
            # Update candidate's marginal only against the newly selected feature
            # Normalize Z once to (n_samples, n_targets)
            z = norm_feature_values[last_selected_name]

            # Pre-normalize X for all viable candidates once (compact)
            cand_order = list(viable_features)
            x_stack = np.stack([norm_feature_values[name] for name in cand_order], axis=0)  # (C, N, T)

            # Batch compute sxz for all candidates: sxz[c, t] = sum_n X[c, n, t] * Z[n, t]
            # Z[None, :, :] adds a candidate axis for broadcasting: Z -> (1, N, T)
            xz = x_stack * z[None, :, :]
            sxz_matrix = xz.sum(axis=1, dtype=np.float64)  # (C, T)

            # Update scores and remove features that fall below threshold
            features_to_remove = []
            for idx, cand_name in enumerate(cand_order):
                sxz_vec = sxz_matrix[idx]
                pairwise_score = self._pairwise_feature_score(
                    stats_by_name[cand_name], stats_by_name[last_selected_name], baseline_stats, sxz_vec
                )
                candidate_marginal_scores[cand_name] = min(candidate_marginal_scores[cand_name], pairwise_score)
                
                # Mark features that fall below threshold for removal
                if candidate_marginal_scores[cand_name] < self.min_marginal_reduction_threshold:
                    features_to_remove.append(cand_name)
            
            # Remove features that no longer meet the threshold
            for feature_name in features_to_remove:
                viable_features.remove(feature_name)
            
            # If no viable features remain, return current selection
            if not viable_features:
                return selected_feature_names, feature_scores

            # Select best remaining viable feature
            best_feature = max(viable_features, key=lambda name: (candidate_marginal_scores[name], name))
            best_score = candidate_marginal_scores[best_feature]
            
            selected_feature_names.append(best_feature)
            feature_scores[best_feature] = best_score
            viable_features.remove(best_feature)

        return selected_feature_names, feature_scores

    @staticmethod
    def _lin_regression_2variables_with_sums(sx: float, sz: float, sy: float, sx2: float, sz2: float,
                                             sxz: float, sxy: float, szy: float, n: float, sy2: float) -> float:
        """Closed-form two-variable linear regression with intercept using sums.
        Returns ((a, b, c), sse).
        """
        # (transpose(X)*X)^-1 * transpose(X)*Y
        a, b, c = LinearPairWiseFeatureSelector._solve_3x3_system(sx2, sxz, sx, sxz, sz2, sz, sx, sz, n, sxy, szy, sy)

        # Calculate SSE: e = a*a*sx2 + 2*a*b*sxz + 2*a*c*sx - 2*a*sxy + b*b*sz2 + 2*b*c*sz - 2*b*szy + n*c*c - 2*c*sy + sy2
        e = (a * a * sx2 + 2 * a * b * sxz + 2 * a * c * sx - 2 * a * sxy +
             b * b * sz2 + 2 * b * c * sz - 2 * b * szy +
             n * c * c - 2 * c * sy + sy2)

        return e

    def _lin_regression_1variable_with_sums(self, sx: float, sy: float, sx2: float, sxy: float, n: float, sy2: float) -> float:
        """Closed-form single-variable linear regression with intercept using sums.
        Returns ((a, c), sse).
        """
        a, c = self._solve_2x2_system(sx2, sx, sx, n, sxy, sy)
        e = (a * a * sx2 + 2 * a * c * sx - 2 * a * sxy + n * c * c - 2 * c * sy + sy2)
        return e

    @staticmethod
    def _calculate_baseline_statistics(target: np.ndarray) -> TargetStats:
        """Calculate baseline target statistics."""
        n_samples = float(target.shape[0])
        sy = target.sum(axis=0)
        sy2 = (target * target).sum(axis=0)
        prior_sses = sy2 - (sy * sy / n_samples)
        stdevs = np.sqrt(prior_sses / n_samples)
        return TargetStats(sy=sy, sy2=sy2, priorsses=prior_sses, stdevs=stdevs, n_samples=n_samples)

    def _calculate_feature_statistics(self, feature_values: np.ndarray, target: np.ndarray) -> FeatureTargetStats:
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
            sses[i] = self._lin_regression_1variable_with_sums(
                float(sx_arr[i]), float(sy[i]), float(sx2_arr[i]), float(sxy[i]), n_samples, float(sy2[i])
            )

        return FeatureTargetStats(sx=sx_arr, sx2=sx2_arr, sxy=sxy, sses=sses)
