"""Universal feature evaluators that can handle any feature type.

These evaluators serve as fallbacks when no more specific evaluator is available.
They work by calling the feature's own evaluate methods, which is less efficient
than specialized batch evaluators but ensures all features can be evaluated.
"""
import asyncio
from collections.abc import Sequence
from typing import Self, cast, override

import polars as pl
from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.feature.base import Feature, SyncFeature
from agentune.analyze.feature.eval.base import FeatureEvaluator, SyncFeatureEvaluator


@frozen
class UniversalSyncFeatureEvaluator(SyncFeatureEvaluator):
    """Universal evaluator for sync features using their batch_evaluate methods."""
    
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
    def evaluate(self, dataset: Dataset,
                 conn: DuckDBPyConnection) -> Dataset:
        new_series = [feature.evaluate_batch_safe(dataset, conn) for feature in self.features]
        new_cols = tuple(Field(feature.name, feature.dtype) for feature in self.features)
        return Dataset(Schema(new_cols), pl.DataFrame({col.name: series for col, series in zip(new_cols, new_series, strict=True)}))


@frozen    
class UniversalAsyncFeatureEvaluator(FeatureEvaluator):
    """Universal evaluator for async features using their aevaluate_batch methods."""
    
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
    async def aevaluate(self, dataset: Dataset,
                        conn: DuckDBPyConnection) -> Dataset:
        new_series = await asyncio.gather(*[feature.aevaluate_batch_safe(dataset, conn) for feature in self.features])
        new_cols = tuple(Field(feature.name, feature.dtype) for feature in self.features)
        return Dataset(Schema(new_cols), pl.DataFrame({col.name: series for col, series in zip(new_cols, new_series, strict=True)}))
