# Toy implementations of all unit types

import asyncio
import math
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Self, override

import polars as pl
from attrs import field, frozen
from duckdb import DuckDBPyConnection

import agentune.analyze.util.copy
from agentune.analyze.context.base import ContextDefinition, TablesWithContextDefinitions
from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.feature.base import Feature, FloatFeature, Regression, SyncFeature
from agentune.analyze.feature.describe.base import FeatureDescriber, SyncFeatureDescriber
from agentune.analyze.feature.eval.base import FeatureEvaluator, SyncFeatureEvaluator
from agentune.analyze.feature.gen.base import (
    FeatureGenerator,
    FeatureTransformer,
    SyncFeatureGenerator,
    SyncFeatureTransformer,
)
from agentune.analyze.feature.select.base import FeatureSelector, SyncFeatureSelector
from agentune.analyze.feature.stats.base import FeatureStats, FeatureWithFullStats
from agentune.analyze.feature.stats.stats import (
    BooleanFeatureStats,
    CategoricalFeatureStats,
    NumericFeatureStats,
)


@frozen
class ToySyncFeature(SyncFeature, FloatFeature):
    """Adds two float columns together."""

    col1: str
    col2: str
    name: str
    description: str
    code: str
    
    @property
    @override
    def params(self) -> Schema: 
        return Schema((Field(self.col1, types.float64), Field(self.col2, types.float64), ))
    
    @property
    @override
    def context_tables(self) -> list[DuckdbTable]:
        return []
    
    @property
    @override
    def context_objects(self) -> list[ContextDefinition]:
        return []
        
    @override
    def evaluate(self, args: tuple[Any, ...], contexts: TablesWithContextDefinitions,
                 conn: DuckDBPyConnection) -> float:
        return args[0] + args[1]
    
    @override
    def evaluate_batch(self, input: Dataset, contexts: TablesWithContextDefinitions,
                       conn: DuckDBPyConnection) -> pl.Series:
        return input.data.get_column(self.col1) + input.data.get_column(self.col2)
    
@frozen
class ToyAsyncFeature(FloatFeature):
    """Adds two float columns together."""

    col1: str
    col2: str
    name: str
    description: str
    code: str

    @property
    @override
    def params(self) -> Schema: 
        return Schema((Field(self.col1, types.float64), Field(self.col2, types.float64), ))
    
    @property
    @override
    def context_tables(self) -> list[DuckdbTable]:
        return []
    
    @property
    @override
    def context_objects(self) -> list[ContextDefinition]:
        return []
        
    @override
    async def aevaluate(self, args: tuple[Any, ...], contexts: TablesWithContextDefinitions,
                        conn: DuckDBPyConnection) -> float:
        return args[0] + args[1]
    
    @override
    async def aevaluate_batch(self, input: Dataset, contexts: TablesWithContextDefinitions,
                              conn: DuckDBPyConnection) -> pl.Series:
        return input.data.get_column(self.col1) + input.data.get_column(self.col2)
    
class ToySyncFeatureGenerator(SyncFeatureGenerator[ToySyncFeature]):
    @override
    def generate(self, input: Dataset, contexts: TablesWithContextDefinitions) -> Iterator[ToySyncFeature]: 
        for col1 in input.schema.cols:
            for col2 in input.schema.cols:
                if col1 != col2 and col1.dtype == types.float64 and col2.dtype == types.float64:
                    yield ToySyncFeature(col1.name, col2.name, f'{col1.name} + {col2.name}', 
                                         f'Adds {col1.name} and {col2.name}', f'{col1.name} + {col2.name}')

class ToyAsyncFeatureGenerator(FeatureGenerator[ToyAsyncFeature]):
    @override
    async def agenerate(self, input: Dataset, contexts: TablesWithContextDefinitions) -> AsyncIterator[ToyAsyncFeature]:
        for col1 in input.schema.cols:
            for col2 in input.schema.cols:
                if col1 != col2 and col1.dtype == types.float64 and col2.dtype == types.float64:
                    await asyncio.sleep(0)
                    yield ToyAsyncFeature(col1.name, col2.name, f'{col1.name} + {col2.name}', 
                                          f'Adds {col1.name} and {col2.name}', f'{col1.name} + {col2.name}')

class ToySyncFeatureTransformer(SyncFeatureTransformer[Feature, Feature]):
    @override
    def transform(self, input: Dataset, contexts: TablesWithContextDefinitions, feature: Feature) -> list[Feature]:
        if hash(str(feature)) < 2**16:
            return []
        else:
            return [agentune.analyze.util.copy.replace(feature, name=f'{feature.name}, transformed')]

class ToyAsyncFeatureTransformer(FeatureTransformer[Feature, Feature]):
    @override
    async def atransform(self, input: Dataset, contexts: TablesWithContextDefinitions, feature: Feature) -> list[Feature]:
        await asyncio.sleep(0)
        if hash(str(feature)) < 2**16:
            return []
        else:
            return [agentune.analyze.util.copy.replace(feature, name=f'{feature.name}, transformed')]

class ToySyncFeatureDescriber(SyncFeatureDescriber[Feature]):
    @override
    def description(self, feature: Feature) -> str:
        return f'{feature.name} described'

class ToyAsyncFeatureDescriber(FeatureDescriber[Feature]):
    @override
    async def adescription(self, feature: Feature) -> str:
        await asyncio.sleep(0)
        return f'{feature.name} described'
    
@frozen
class ToySyncFeatureEvaluator(SyncFeatureEvaluator[SyncFeature]):
    features: tuple[SyncFeature, ...]

    @override
    @classmethod
    def supports_feature(cls, feature: Feature) -> bool:
        return isinstance(feature, SyncFeature)

    @override 
    @classmethod
    def for_features(cls, features: Sequence[SyncFeature]) -> Self:
        return cls(tuple(features))

    @override
    def evaluate(self, dataset: Dataset, contexts: TablesWithContextDefinitions, 
                 conn: DuckDBPyConnection, include_originals: bool) -> Dataset:
        new_series = [feature.evaluate_batch(dataset, contexts, conn) for feature in self.features]
        new_cols = tuple(Field(feature.name, feature.dtype) for feature in self.features)
        if include_originals:
            df = dataset.data.with_columns(**{col.name: series for col, series in zip(new_cols, new_series, strict=True)})
            schema = Schema(dataset.schema.cols + new_cols)
            return Dataset(schema, df)
        else:
            return Dataset(Schema(tuple(new_cols)), pl.DataFrame({col.name: series for col, series in zip(new_cols, new_series, strict=True)}))

@frozen    
class ToyAsyncFeatureEvaluator(FeatureEvaluator[Feature]):
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
    async def aevaluate(self, dataset: Dataset, contexts: TablesWithContextDefinitions, 
                        conn: DuckDBPyConnection, include_originals: bool) -> Dataset:
        new_series = [await feature.aevaluate_batch(dataset, contexts, conn) for feature in self.features]
        new_cols = tuple(Field(feature.name, feature.dtype) for feature in self.features)
        if include_originals:
            df = dataset.data.with_columns(**{col.name: series for col, series in zip(new_cols, new_series, strict=True)})
            schema = Schema(dataset.schema.cols + new_cols)
            return Dataset(schema, df)
        else:
            return Dataset(Schema(new_cols), pl.DataFrame({col.name: series for col, series in zip(new_cols, new_series, strict=True)}))


@frozen
class ToySyncFeatureSelector(SyncFeatureSelector[Feature, Regression]):
    features: list[FeatureWithFullStats[Feature, Regression]] = field(factory=list)

    @override
    def add_feature(self, feature_with_stats: FeatureWithFullStats[Feature, Regression]) -> None:
        self.features.append(feature_with_stats)
    
    @staticmethod
    def some_metric(stats: FeatureStats) -> float:
        if isinstance(stats, NumericFeatureStats):
            return stats.mean
        elif isinstance(stats, CategoricalFeatureStats):
            return stats.unique_count
        elif isinstance(stats, BooleanFeatureStats):
            return stats.coverage
        else:
            raise TypeError(f'Unknown feature stats type: {type(stats)}')

    @override
    def select_final_features(self) -> list[FeatureWithFullStats[Feature, Regression]]:
        average_metric = math.nan if len(self.features) == 0 \
            else sum(ToySyncFeatureSelector.some_metric(x.stats.feature) for x in self.features) / len(self.features)
        return [x for x in self.features if ToySyncFeatureSelector.some_metric(x.stats.feature) >= average_metric]

@frozen
class ToyAsyncFeatureSelector(FeatureSelector[Feature, Regression]):
    features: list[FeatureWithFullStats[Feature, Regression]] = field(factory=list)

    @override
    async def aadd_feature(self, feature_with_stats: FeatureWithFullStats[Feature, Regression]) -> None:
        await asyncio.sleep(0)
        self.features.append(feature_with_stats)

    @override
    async def aselect_final_features(self) -> list[FeatureWithFullStats[Feature, Regression]]:
        await asyncio.sleep(0)
        average_metric = math.nan if len(self.features) == 0 \
            else sum(ToySyncFeatureSelector.some_metric(x.stats.feature) for x in self.features) / len(self.features)
        return [x for x in self.features if ToySyncFeatureSelector.some_metric(x.stats.feature) >= average_metric]
