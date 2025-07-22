import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import override

from agentune.analyze.core.dataset import Dataset, DatasetSource
from agentune.analyze.core.schema import Schema


class Classifier[T](ABC):
    """A classifier's input matches its input_schema. Unexpected input columns should be ignored.

    The output is described by output_schema. It has one float64 column per class with its predicted probability, 
    named probability_$class, in the same order as self.classes (so you don't have to rely on column names);
    and one last column of type T named "predicted" with the class value that has the highest probability.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def classes(self) -> Sequence[T]: ...
    
    @property
    @abstractmethod
    def input_schema(self) -> Schema: ...

    @property
    @abstractmethod
    def output_schema(self) -> Schema: ...

    @abstractmethod
    async def aclassify(self, dataset: Dataset) -> Dataset: ...

class SyncClassifier[T](Classifier[T]):
    @abstractmethod
    def classify(self, dataset: Dataset) -> Dataset: ...

    @override
    async def aclassify(self, dataset: Dataset) -> Dataset: 
        return await asyncio.to_thread(self.classify, dataset.copy_to_thread())

class Regressor(ABC):
    """A regressor's input matches its input_schema. Unexpected input columns should be ignored.
    
    The output has a single float64 column named "predicted".
    """

    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @property
    @abstractmethod
    def input_schema(self) -> Schema: ...

    @abstractmethod
    async def aestimate(self, dataset: Dataset) -> Dataset: ...

class SyncRegressor(Regressor):
    @abstractmethod
    def estimate(self, dataset: Dataset) -> Dataset: ...

    @override
    async def aestimate(self, dataset: Dataset) -> Dataset: 
        return await asyncio.to_thread(self.estimate, dataset.copy_to_thread())

class ClassifierTrainer[T](ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def atrain(self, classes: Sequence[T], data: DatasetSource, target_col: str, weight_col: str) -> Classifier[T]: ...
    
class SyncClassifierTrainer[T](ClassifierTrainer[T]):
    @abstractmethod
    def train(self, classes: Sequence[T], data: DatasetSource, target_col: str, weight_col: str) -> Classifier[T]: ...

    @override
    async def atrain(self, classes: Sequence[T], data: DatasetSource, target_col: str, weight_col: str) -> Classifier[T]: 
        return await asyncio.to_thread(self.train, classes, data.copy_to_thread(), target_col, weight_col)

class RegressorTrainer(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def atrain(self, data: DatasetSource, target_col: str, weight_col: str) -> Regressor: ...
    
class SyncRegressorTrainer(RegressorTrainer):
    @abstractmethod
    def train(self, data: DatasetSource, target_col: str, weight_col: str) -> Regressor: ...

    @override
    async def atrain(self, data: DatasetSource, target_col: str, weight_col: str) -> Regressor: 
        return await asyncio.to_thread(self.train, data.copy_to_thread(), target_col, weight_col)

