import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import override

from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.core.dataset import Dataset
from agentune.analyze.feature.base import Feature
from agentune.analyze.feature.problem import Problem
from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.analyze.util.queue import Queue


@frozen
class GeneratedFeature[F: Feature]:
    feature: F
    has_good_defaults: bool
    # If False, the feature search will replace the feature's default values using some default logic;
    # if True, it will leave the existing defaults in place.


class FeatureGenerator[F: Feature](ABC): 
    @abstractmethod
    def agenerate(self, feature_search: Dataset, problem: Problem, join_strategies: TablesWithJoinStrategies,
                  conn: DuckDBPyConnection) -> AsyncIterator[GeneratedFeature[F]]: ...

# Note that a SyncFeatureGenerator is a generator that operates synchronously, not a generator that generates SyncFeatures.
class SyncFeatureGenerator[F: Feature](FeatureGenerator[F]):
    @abstractmethod
    def generate(self, feature_search: Dataset, problem: Problem, join_strategies: TablesWithJoinStrategies,
                 conn: DuckDBPyConnection) -> Iterator[GeneratedFeature[F]]: ...

    @override
    async def agenerate(self, feature_search: Dataset, problem: Problem, join_strategies: TablesWithJoinStrategies,
                        conn: DuckDBPyConnection) -> AsyncIterator[GeneratedFeature[F]]:
        queue = Queue[GeneratedFeature[F]](1)
        with conn.cursor() as cursor:
            task = asyncio.create_task(asyncio.to_thread(
                lambda: queue.consume(self.generate(feature_search.copy_to_thread(), problem, join_strategies, cursor))))
            async for item in queue:
                yield item
            await task


class FeatureTransformer[FA: Feature, FB: Feature](ABC): 
    """Builds new features on top of an existing one.
    For example, a FeatureTransformer[float, bool] which fits cutoffs and ranges to a numeric feature.
    """
    @abstractmethod
    async def atransform(self, feature_search: Dataset, problem: Problem, join_strategies: TablesWithJoinStrategies,
                         conn: DuckDBPyConnection, feature: FA) -> Sequence[FB] : ...


class SyncFeatureTransformer[FA: Feature, FB: Feature](FeatureTransformer[FA, FB]):
    @abstractmethod
    def transform(self, feature_search: Dataset, problem: Problem, join_strategies: TablesWithJoinStrategies,
                  conn: DuckDBPyConnection, feature: FA) -> Sequence[FB] : ...

    @override
    async def atransform(self, feature_search: Dataset, problem: Problem, join_strategies: TablesWithJoinStrategies,
                         conn: DuckDBPyConnection, feature: FA) -> Sequence[FB]:
        with conn.cursor() as cursor:
            return await asyncio.to_thread(self.transform, feature_search.copy_to_thread(), problem,
                                           join_strategies, cursor, feature)

