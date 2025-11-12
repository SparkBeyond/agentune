import math
from collections.abc import Sequence
from typing import override

import lightgbm as lgb
import numpy as np
import polars as pl
from duckdb import DuckDBPyConnection

from agentune.analyze.core.dataset import DatasetSource
from agentune.analyze.feature.base import Feature
from agentune.analyze.feature.problem import Problem, TargetKind
from agentune.analyze.feature.select.base import SyncEnrichedFeatureSelector

# Dataset size thresholds for parameter tuning
SMALL_DATASET_THRESHOLD = 1000
MEDIUM_DATASET_THRESHOLD = 5000

def _get_lightgbm_params(n_rows: int) -> dict:
    """Get LightGBM parameters based on dataset size."""
    if n_rows < SMALL_DATASET_THRESHOLD:  # TODO: We would like x samples in each leaf, not a fixed number of rows, try to adjust to that.
        max_depth = 3
        base_estimators = 200
    elif n_rows < MEDIUM_DATASET_THRESHOLD:
        max_depth = 5
        base_estimators = 300
    else:
        max_depth = 7
        base_estimators = 500

    feat_adj = 1.0 + 0.2 * math.log2(max(2, n_rows))
    n_estimators = int(min(800, max(50, round(base_estimators * feat_adj))))

    min_child_samples = max(10, n_rows // max(1, (2 ** max(0, max_depth - 1))))

    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_child_samples': int(min_child_samples),
        'learning_rate': 0.05,
        'reg_lambda': 1.0,
        'verbosity': -1,
        'random_state': 42,
    }
    return params


class LightGBMFeatureSelector(SyncEnrichedFeatureSelector):
    """Select top_k features using a single LightGBM model."""

    def __init__(self) -> None:
        """Initialize with unified constructor. Data/columns are passed to select_features()."""
        self.importance_type = 'gain'
        # Model-agnostic selector state (mirror LinearPairWise)
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
        """Select features using the enriched API, matching the base interface signature."""
        df = enriched_data.to_dataset(conn).data
        feature_cols = [f.name for f in features]
        selected_names = self._select_features_df(df, problem.target_column.name, problem.target_kind, feature_cols, max_features_to_select)
        return [f for f in features if f.name in selected_names]

    def _select_features_df(self, data: pl.DataFrame, target_col: str, target_kind: TargetKind, feature_cols: list[str],
                            max_features_to_select: int) -> list[str]:
        """Select max_features_to_select features using a single LightGBM model (DataFrame API)."""
        if not feature_cols:
            self.final_importances_ = {'feature': [], 'importance': []}
            self._selected_feature_names = []
            return []

        x_train = data.select(feature_cols)
        target_type = pl.Int64 if target_kind == 'classification' else pl.Float64
        y_train = data.select(target_col).to_series().cast(target_type)

        params = _get_lightgbm_params(x_train.height)
        # 1) Fit initial model for feature importance
        initial_model = (
            lgb.LGBMClassifier(**params)
            if target_kind == 'classification'
            else lgb.LGBMRegressor(**params)
        )
        initial_model.fit(x_train.to_numpy(), y_train.to_numpy())
        all_importances = initial_model.booster_.feature_importance(importance_type=self.importance_type)

        # Rank and select max_features_to_select features
        sorted_indices = np.argsort(all_importances)[::-1]
        top_k_indices = sorted_indices[: max_features_to_select]
        selected_features = [feature_cols[i] for i in top_k_indices]

        # 2) Refit FINAL model on the selected features using LightGBM for proper importances
        x_train_topk = data.select(selected_features)
        x_train_topk_np = x_train_topk.to_numpy()
        y_train_np = y_train.to_numpy()

        # Fit final LightGBM model for importances
        params = _get_lightgbm_params(x_train_topk.height)
        if target_kind == 'classification':
            final_lgb_model = lgb.LGBMClassifier(**params)
        else:
            final_lgb_model = lgb.LGBMRegressor(**params)
        
        final_lgb_model.fit(x_train_topk_np, y_train_np)
        final_importances = final_lgb_model.booster_.feature_importance(importance_type=self.importance_type)
        feature_scores = dict(zip(selected_features, final_importances, strict=False))
        sorted_feature_names = sorted(feature_scores.keys(), key=lambda name: feature_scores[name],reverse=True)

        self.final_importances_ = {
            'feature': sorted_feature_names,
            'importance': [feature_scores[name] for name in sorted_feature_names]
        }

        self._selected_feature_names = self.final_importances_['feature']
        return self._selected_feature_names
