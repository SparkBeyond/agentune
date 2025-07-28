import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import override

from agentune.analyze.feature.base import Feature, TargetKind
from agentune.analyze.feature.stats.base import FeatureWithFullStats

# TODO add 'EnrichedFeatureSelector' that takes as input the full enriched output of the each feature on the feature search dataset,
#  and the target column, and also the stats.
#  Try to make an API that stores the data in duckdb and doesn't just pass Series around.

class FeatureSelector[F: Feature, T: TargetKind](ABC): 
    @abstractmethod
    async def aadd_feature(self, feature_with_stats: FeatureWithFullStats[F, T]) -> None: ...

    @abstractmethod
    async def aselect_final_features(self) -> Sequence[FeatureWithFullStats[F, T]]: ...

class SyncFeatureSelector[F: Feature, T: TargetKind](FeatureSelector[F, T]):
    @abstractmethod
    def add_feature(self, feature_with_stats: FeatureWithFullStats[F, T]) -> None: ...

    @override
    async def aadd_feature(self, feature_with_stats: FeatureWithFullStats[F, T]) -> None:
        await asyncio.to_thread(self.add_feature, feature_with_stats)

    @abstractmethod
    def select_final_features(self) -> Sequence[FeatureWithFullStats[F, T]]: ...

    @override
    async def aselect_final_features(self) -> Sequence[FeatureWithFullStats[F, T]]:
        return await asyncio.to_thread(self.select_final_features)

