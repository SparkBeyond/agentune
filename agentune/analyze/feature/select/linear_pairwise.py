import logging
from collections.abc import Sequence
from typing import override

import numpy as np
from duckdb import DuckDBPyConnection

from agentune.analyze.core.dataset import DatasetSource
from agentune.analyze.feature.base import Feature
from agentune.analyze.feature.problem import Problem
from agentune.analyze.feature.select.base import SyncEnrichedFeatureSelector
from agentune.analyze.feature.util import substitute_default_values
from agentune.analyze.util.feature_sse_reduction import (
    FeatureTargetStats,
    TargetStats,
    calculate_feature_statistics,
    prepare_sse_data,
    single_feature_score,
)

logger = logging.getLogger(__name__)

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
        min_marginal_reduction_threshold: float = 1e-8
    ):
        """Initialize selector.

        Args:
            min_marginal_reduction_threshold: Minimum marginal error reduction required to add a feature.
                Features with lower marginal improvement will not be selected.
        """
        self.min_marginal_reduction_threshold = min_marginal_reduction_threshold

        # No internal model; selector is model-agnostic
        self.final_importances_: dict[str, list] | None = None
        self._selected_feature_names: list[str] | None = None

    @override
    def select_features(
        self,
        features: Sequence[Feature],
        max_features_to_select: int,
        enriched_data: DatasetSource,
        problem: Problem,
        conn: DuckDBPyConnection,
    ) -> Sequence[Feature]:
        """Select features using the enriched API, matching the base interface signature.

        Args:
            features: Sequence of Feature definitions.
            enriched_data: DatasetSource containing materialized feature columns and target.
            problem: Problem definition containing target column information.
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

        target, feature_values, baseline_stats = prepare_sse_data(features, df_pl, target_column)

        remaining_feature_names = list(feature_values.keys())

        if not remaining_feature_names:
            self.final_importances_ = {'feature': [], 'importance': []}
            return []

        selected_feature_names, feature_scores = self._select_best_features(feature_values, target, baseline_stats, max_features_to_select)
        
        # Return features in selection order (not importance order)
        feature_by_name = {f.name: f for f in features}
        selected_features = [feature_by_name[name] for name in selected_feature_names]

        # Store final importances in selection order to match returned features
        self.final_importances_ = {
            'feature': selected_feature_names,  # Already in selection order
            'importance': feature_scores
        }

        return selected_features


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
    def _preprocess_data_array(data: np.ndarray) -> np.ndarray:
        """Validate data arrays contain only finite values; rely on upstream imputation."""
        if not np.all(np.isfinite(data)):
            raise ValueError('an infinite value was passed')
        return data


    @staticmethod
    def _normalize_to_targets(values: np.ndarray, n_targets: int) -> np.ndarray:
        """Normalize 1D or single-column arrays to (n_samples, n_targets); pass through multi-target arrays."""
        x = values[:, None] if values.ndim == 1 else values
        return np.repeat(x, n_targets, axis=1) if x.shape[1] == 1 else x

    def _select_best_features(self, feature_values: dict[str, np.ndarray], target: np.ndarray, baseline_stats: TargetStats,
                              max_features_to_select: int) -> tuple[list[str], list[float]]:
        """Two-phase feature selection: single feature scoring, then pairwise marginalization."""
        remaining_feature_names = list(feature_values.keys())
        selected_feature_names: list[str] = []
        feature_scores: list[float] = []

        # Precompute per-feature statistics once for reuse in scoring (local per-run cache)
        stats_by_name: dict[str, FeatureTargetStats] = {
            fname: calculate_feature_statistics(vals, target) for fname, vals in feature_values.items()
        }

        # Phase 1: single feature scoring and initial filtering
        candidate_marginal_scores: dict[str, float] = {}
        for f_name in remaining_feature_names:
            sse_reduction, _ = single_feature_score(stats_by_name[f_name], baseline_stats)
            candidate_marginal_scores[f_name] = sse_reduction
        if not remaining_feature_names:
            self.final_importances_ = {'feature': [], 'importance': []}
            return [], []

        # Filter out features that don't meet the threshold (they can never improve)
        viable_features = [
            f_name for f_name in remaining_feature_names
            if candidate_marginal_scores[f_name] > 0.0 and candidate_marginal_scores[f_name] >= self.min_marginal_reduction_threshold
        ]

        if not viable_features:
            # No features meet the threshold, return empty selection
            return [], []

        # Select best first feature from viable candidates
        best_first_feature = max(viable_features, key=lambda name: (candidate_marginal_scores[name], name))
        best_first_score = candidate_marginal_scores[best_first_feature]

        selected_feature_names.append(best_first_feature)
        feature_scores.append(best_first_score)
        viable_features.remove(best_first_feature)

        # Phase 2: iteratively add features using pairwise scoring
        # Precompute target-shape constants and per-feature normalization outside the loop
        n_targets = target.shape[1]
        norm_feature_values: dict[str, np.ndarray] = {
            name: self._normalize_to_targets(vals, n_targets) for name, vals in feature_values.items()
        }
        while len(selected_feature_names) < max_features_to_select and viable_features:
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
            feature_scores.append(best_score)
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

