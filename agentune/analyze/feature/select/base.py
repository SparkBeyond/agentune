import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import override

from duckdb import DuckDBPyConnection

from agentune.analyze.core.dataset import DatasetSource
from agentune.analyze.feature.base import Feature
from agentune.analyze.feature.problem import Problem
from agentune.analyze.feature.stats.base import FeatureWithFullStats


class FeatureSelector(ABC):
    """Select a subset of the candidate features.

    The feature stats are calculated on the feature eval dataset.
    """

    @abstractmethod
    async def aadd_feature(self, feature_with_stats: FeatureWithFullStats) -> None: ...

    @abstractmethod
    async def aselect_final_features(self, problem: Problem) -> Sequence[FeatureWithFullStats]: ...

class SyncFeatureSelector(FeatureSelector):
    @abstractmethod
    def add_feature(self, feature_with_stats: FeatureWithFullStats) -> None: ...

    @override
    async def aadd_feature(self, feature_with_stats: FeatureWithFullStats) -> None:
        await asyncio.to_thread(self.add_feature, feature_with_stats)

    @abstractmethod
    def select_final_features(self, problem: Problem) -> Sequence[FeatureWithFullStats]: ...

    @override
    async def aselect_final_features(self, problem: Problem) -> Sequence[FeatureWithFullStats]:
        return await asyncio.to_thread(self.select_final_features, problem)

class EnrichedFeatureSelector(ABC):
    """A FeatureSelector that requires the entire enriched feature output, not only statistics.

    It operates on all features at once, which assumes there are relatively few features.

    Although this interface could extend (and easily implement) the FeatureSelector interface,
    it deliberately doesn't; code should be aware which kind of selector it's using,
    because it matters a lot to resource management.
    """

    @abstractmethod
    async def aselect_features(self, features: Sequence[Feature],
                               enriched_data: DatasetSource, problem: Problem,
                               conn: DuckDBPyConnection) -> Sequence[Feature]:
        """enriched_data contains a column for each feature; the column's name is the feature.name.

        It also contains the target column.
        """
        ...

class SyncEnrichedFeatureSelector(EnrichedFeatureSelector):
    @abstractmethod
    def select_features(self, features: Sequence[Feature],
                        enriched_data: DatasetSource, problem: Problem,
                        conn: DuckDBPyConnection) -> Sequence[Feature]: ...

    @override
    async def aselect_features(self, features: Sequence[Feature],
                               enriched_data: DatasetSource, problem: Problem,
                               conn: DuckDBPyConnection) -> Sequence[Feature]:
        with conn.cursor() as cursor:
            return await asyncio.to_thread(self.select_features, features, enriched_data, problem, cursor)
