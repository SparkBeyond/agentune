"""Universal feature computers that can handle any feature type.

These feature computers serve as fallbacks when no more specific feature computer is available.
They work by calling the feature's own compute methods, which is less efficient
than specialized batch feature computers but ensures all features can be computed.
"""
import asyncio
import logging
from collections.abc import Sequence
from typing import Self, cast, override

import polars as pl
from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.feature.base import Feature, SyncFeature
from agentune.analyze.feature.compute.base import FeatureComputer, SyncFeatureComputer
from agentune.analyze.progress.base import ProgressStage, stage_scope

_logger = logging.getLogger(__name__)

@frozen
class UniversalSyncFeatureComputer(SyncFeatureComputer):
    """Universal feature computer for sync features using their batch_compute methods."""
    
    features: tuple[SyncFeature, ...]

    @override
    @classmethod
    def supports_feature(cls, feature: Feature) -> bool:
        return isinstance(feature, SyncFeature)

    @override 
    @classmethod
    def for_features(cls, features: Sequence[Feature]) -> Self:
        return cls(cast(tuple[SyncFeature, ...], tuple(features)))

    @override
    def compute(self, dataset: Dataset, conn: DuckDBPyConnection,
                cells_progress: ProgressStage | None = None) -> Dataset:
        if cells_progress is None:
            with stage_scope(f'Compute cells (features*rows) on {dataset.height} rows', 0, len(self.features) * dataset.height) as new_cells_progress:
                return self.compute(dataset, conn, new_cells_progress)
        else:
            def compute_feature(feature: SyncFeature) -> pl.Series:
                result = feature.compute_batch_safe(dataset, conn)
                cells_progress.increment_count(dataset.height)
                return result
            new_series = [compute_feature(feature) for feature in self.features]
            new_cols = tuple(Field(feature.name, feature.dtype) for feature in self.features)
            return Dataset(Schema(new_cols), pl.DataFrame({col.name: series for col, series in zip(new_cols, new_series, strict=True)}))


@frozen    
class UniversalAsyncFeatureComputer(FeatureComputer):
    """Universal feature computer for async features using their acompute_batch methods."""
    
    features: tuple[Feature, ...]

    @override
    @classmethod
    def supports_feature(cls, feature: Feature) -> bool:
        return not isinstance(feature, SyncFeature)

    @override 
    @classmethod
    def for_features(cls, features: Sequence[Feature]) -> Self:
        return cls(tuple(features))
    
    @override
    async def acompute(self, dataset: Dataset, conn: DuckDBPyConnection,
                       cells_progress: ProgressStage | None = None) -> Dataset:
        if cells_progress is None:
            with stage_scope(f'Compute cells (features*rows) on {dataset.height} rows', 0, len(self.features) * dataset.height) as new_cells_progress:
                return await self.acompute(dataset, conn, new_cells_progress)
        else:
            async def acompute_feature(feature: Feature) -> pl.Series:
                result = await feature.acompute_batch_safe(dataset, conn)
                cells_progress.increment_count(dataset.height)
                return result
            new_series = await asyncio.gather(*[acompute_feature(feature) for feature in self.features])
            new_cols = tuple(Field(feature.name, feature.dtype) for feature in self.features)
            return Dataset(Schema(new_cols), pl.DataFrame({col.name: series for col, series in zip(new_cols, new_series, strict=True)}))
