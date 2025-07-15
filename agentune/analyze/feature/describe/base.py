import asyncio
from abc import ABC, abstractmethod
from typing import override

from agentune.analyze.feature.base import Feature
from agentune.analyze.util.copy import replace


class FeatureDescriber[F: Feature](ABC): 
    """Improves on a feature's description.
    
    This used to be called 'feature humanization'.
    """

    @abstractmethod
    async def adescription(self, feature: F) -> str: 
        """Returns a new feature.description."""
        ...

    async def adescribe(self, feature: F) -> F:
        return replace(feature, description=await self.adescription(feature))


class SyncFeatureDescriber[F: Feature](FeatureDescriber[F]):
    @abstractmethod
    def description(self, feature: F) -> str: 
        """Returns a new feature.description."""
        ...

    def describe(self, feature: F) -> F:
        return replace(feature, description=self.description(feature))
    
    @override 
    async def adescription(self, feature: F) -> str: 
        return await asyncio.to_thread(self.description, feature)
    
    @override
    async def adescribe(self, feature: F) -> F:
        return await asyncio.to_thread(self.describe, feature)
    
