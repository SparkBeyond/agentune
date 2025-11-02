import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self, override

from duckdb import DuckDBPyConnection

from agentune.analyze.core.dataset import Dataset
from agentune.analyze.feature.base import Feature
from agentune.analyze.progress.base import ProgressStage


class FeatureComputer(ABC):
    """A feature computer can compute many features at once more efficiently than calling each feature's compute method one by one.
    
    This works only for features with particular similiarities: e.g. a group of async features, SQL query features, or AST-based features.
    """

    @classmethod
    @abstractmethod
    def supports_feature(cls, feature: Feature) -> bool:
        """Whether this computer can compute this feature, together with other features for which it returns True,
        more efficiently than computing them one by one (or in parallel in the case of async features).
        """
        raise NotImplementedError # returning ... computes as false in a boolean context

    @classmethod
    @abstractmethod
    def for_features(cls, features: Sequence[Feature]) -> Self: ...

    @property
    @abstractmethod
    def features(self) -> Sequence[Feature]:
        ...

    @abstractmethod
    async def acompute(self, dataset: Dataset, conn: DuckDBPyConnection,
                       cells_progress: ProgressStage | None = None) -> Dataset:
        """Args:
            dataset: includes all columns needed by all the features. Any additional columns must be ignored by the implementation.
            conn: makes available contains data declared in `secondary_tables` or `join_strategies`.
                  Any additional tables or columns must be ignored by the implementation.
            cells_progress: will be used to increment the count of cells (i.e. rows*features) computed.
                            If not given, a new stage will be created for the duration of the call.

        Returns:
            A dataset with a column per feature, named with the feature's name.
        """
        ...

class SyncFeatureComputer(FeatureComputer):
    @abstractmethod
    def compute(self, dataset: Dataset, conn: DuckDBPyConnection,
                cells_progress: ProgressStage | None = None) -> Dataset:
        """See FeatureComputer.acompute for details."""
        ...

    @override
    async def acompute(self, dataset: Dataset, conn: DuckDBPyConnection,
                       cells_progress: ProgressStage | None = None) -> Dataset:
        with conn.cursor() as cursor:
            return await asyncio.to_thread(self.compute, dataset.copy_to_thread(), cursor, cells_progress)
