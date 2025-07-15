import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import override

from agentune.analyze.context.base import TablesWithContextDefinitions
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.feature.base import Feature
from agentune.analyze.util.queue import Queue


class FeatureGenerator[F: Feature](ABC): 
    @abstractmethod
    def agenerate(self, input: Dataset, contexts: TablesWithContextDefinitions) -> AsyncIterator[F]: ...

# Note that a SyncFeatureGenerator is a generator that operates synchronously, not a generator that generates SyncFeatures.
class SyncFeatureGenerator[F: Feature](FeatureGenerator[F]):
    @abstractmethod
    def generate(self, input: Dataset, contexts: TablesWithContextDefinitions) -> Iterator[F]: ...

    @override
    async def agenerate(self, input: Dataset, contexts: TablesWithContextDefinitions) -> AsyncIterator[F]:
        queue = Queue[F](1)
        task = asyncio.to_thread(lambda: queue.consume(self.generate(input.copy_to_thread(), contexts)))
        async for item in queue:
            yield item
        await task
    

class FeatureTransformer[FA: Feature, FB: Feature](ABC): 
    """Builds new features on top of an existing one.
    For example, a FeatureTransformer[float, bool] which fits cutoffs and ranges to a numeric feature.
    """
    @abstractmethod
    async def atransform(self, input: Dataset, contexts: TablesWithContextDefinitions, feature: FA) -> Sequence[FB] : ...


class SyncFeatureTransformer[FA: Feature, FB: Feature](FeatureTransformer[FA, FB]):
    @abstractmethod
    def transform(self, input: Dataset, contexts: TablesWithContextDefinitions, feature: FA) -> Sequence[FB] : ...

    @override
    async def atransform(self, input: Dataset, contexts: TablesWithContextDefinitions, feature: FA) -> Sequence[FB]:
        return await asyncio.to_thread(self.transform, input.copy_to_thread(), contexts, feature)

