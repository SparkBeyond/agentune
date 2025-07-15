import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import override

from agentune.analyze.feature.base import Feature, TargetKind
from agentune.analyze.feature.stats.base import FeatureWithFullStats


class FeatureSelector[F: Feature, T: TargetKind](ABC): 
    @abstractmethod
    async def aadd_feature(self, feature_with_stats: FeatureWithFullStats[F, T]) -> None: ...

    @abstractmethod
    async def aselect_final_features(self) -> Iterable[FeatureWithFullStats[F, T]]: ...

class SyncFeatureSelector[F: Feature, T: TargetKind](FeatureSelector[F, T]):
    @abstractmethod
    def add_feature(self, feature_with_stats: FeatureWithFullStats[F, T]) -> None: ...

    @override
    async def aadd_feature(self, feature_with_stats: FeatureWithFullStats[F, T]) -> None:
        await asyncio.to_thread(self.add_feature, feature_with_stats)

    @abstractmethod
    def select_final_features(self) -> Iterable[FeatureWithFullStats[F, T]]: ...

    @override
    async def aselect_final_features(self) -> Iterable[FeatureWithFullStats[F, T]]:
        return await asyncio.to_thread(self.select_final_features)

