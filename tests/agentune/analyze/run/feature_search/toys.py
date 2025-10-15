# Toy implementations of all unit types

import asyncio
import logging
import math
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, cast, override

import polars as pl
from attrs import field, frozen
from duckdb import DuckDBPyConnection

import agentune.analyze.util.copy
from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.dataset import Dataset, DatasetSource
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.feature.base import (
    BoolFeature,
    CategoricalFeature,
    Feature,
    FloatFeature,
    NumericFeature,
    SyncFloatFeature,
)
from agentune.analyze.feature.describe.base import FeatureDescriber, SyncFeatureDescriber
from agentune.analyze.feature.gen.base import (
    FeatureGenerator,
    FeatureTransformer,
    GeneratedFeature,
    SyncFeatureGenerator,
    SyncFeatureTransformer,
)
from agentune.analyze.feature.problem import Problem
from agentune.analyze.feature.select.base import (
    EnrichedFeatureSelector,
    FeatureSelector,
    SyncEnrichedFeatureSelector,
    SyncFeatureSelector,
)
from agentune.analyze.feature.stats.base import FeatureStats, FeatureWithFullStats
from agentune.analyze.join.base import JoinStrategy, TablesWithJoinStrategies

_logger = logging.getLogger(__name__)


@frozen
class ToySyncFeature(SyncFloatFeature):
    """Adds two float columns together."""

    col1: str
    col2: str
    name: str
    description: str
    technical_description: str

    # Redeclare attributes with defaults
    default_for_missing: float = 0.0
    default_for_nan: float = 0.0
    default_for_infinity: float = 0.0
    default_for_neg_infinity: float = 0.0

    @property
    @override
    def params(self) -> Schema: 
        return Schema((Field(self.col1, types.float64), Field(self.col2, types.float64), ))
    
    @property
    @override
    def secondary_tables(self) -> list[DuckdbTable]:
        return []
    
    @property
    @override
    def join_strategies(self) -> list[JoinStrategy]:
        return []
        
    @override
    def evaluate(self, args: tuple[Any, ...],
                 conn: DuckDBPyConnection) -> float:
        return args[0] + args[1]
    
    @override
    def evaluate_batch(self, input: Dataset,
                       conn: DuckDBPyConnection) -> pl.Series:
        return input.data.get_column(self.col1) + input.data.get_column(self.col2)
    
@frozen
class ToyAsyncFeature(FloatFeature):
    """Adds two float columns together."""

    col1: str
    col2: str
    name: str
    description: str
    technical_description: str

    # Redeclare attributes with defaults
    default_for_missing: float = 0.0
    default_for_nan: float = 0.0
    default_for_infinity: float = 0.0
    default_for_neg_infinity: float = 0.0

    @property
    @override
    def params(self) -> Schema: 
        return Schema((Field(self.col1, types.float64), Field(self.col2, types.float64), ))
    
    @property
    @override
    def secondary_tables(self) -> list[DuckdbTable]:
        return []
    
    @property
    @override
    def join_strategies(self) -> list[JoinStrategy]:
        return []
        
    @override
    async def aevaluate(self, args: tuple[Any, ...],
                        conn: DuckDBPyConnection) -> float:
        return args[0] + args[1]
    
    @override
    async def aevaluate_batch(self, input: Dataset,
                              conn: DuckDBPyConnection) -> pl.Series:
        return input.data.get_column(self.col1) + input.data.get_column(self.col2)
    
class ToySyncFeatureGenerator(SyncFeatureGenerator):
    @override
    def generate(self, feature_search: Dataset, problem: Problem, join_strategies: TablesWithJoinStrategies,
                 conn: DuckDBPyConnection) -> Iterator[GeneratedFeature]:
        for col1 in feature_search.schema.cols:
            for col2 in feature_search.schema.cols:
                if problem.target_column.name not in (col1.name, col2.name) and \
                        col1 != col2 and col1.dtype == types.float64 and col2.dtype == types.float64:
                    feature = ToySyncFeature(col1.name, col2.name, f'Sync {col1.name} + {col2.name}',
                                             f'Adds {col1.name} and {col2.name}', f'{col1.name} + {col2.name}')
                    yield GeneratedFeature(feature, False)

class ToyAsyncFeatureGenerator(FeatureGenerator):
    @override
    async def agenerate(self, feature_search: Dataset, problem: Problem, join_strategies: TablesWithJoinStrategies,
                        conn: DuckDBPyConnection) -> AsyncIterator[GeneratedFeature]:
        for col1 in feature_search.schema.cols:
            for col2 in feature_search.schema.cols:
                if problem.target_column.name not in (col1.name, col2.name) and \
                        col1 != col2 and col1.dtype == types.float64 and col2.dtype == types.float64:
                    await asyncio.sleep(0)
                    feature = ToyAsyncFeature(col1.name, col2.name, f'Async {col1.name} + {col2.name}',
                                              f'Adds {col1.name} and {col2.name}', f'{col1.name} + {col2.name}')
                    yield GeneratedFeature(feature, False)

@frozen
class ToyPrebuiltFeaturesGenerator(SyncFeatureGenerator):
    features: tuple[Feature, ...]

    @override
    def generate(self, feature_search: Dataset, problem: Problem, join_strategies: TablesWithJoinStrategies,
                 conn: DuckDBPyConnection) -> Iterator[GeneratedFeature]:
        for feature in self.features:
            yield GeneratedFeature(feature, True)

class ToySyncFeatureTransformer(SyncFeatureTransformer):
    @override
    def transform(self, feature_search: Dataset, problem: Problem, join_strategies: TablesWithJoinStrategies,
                  conn: DuckDBPyConnection, feature: Feature) -> list[Feature]:
        if hash(str(feature)) < 2**16:
            return []
        else:
            return [agentune.analyze.util.copy.replace(feature, name=f'{feature.name}, transformed')]

class ToyAsyncFeatureTransformer(FeatureTransformer):
    @override
    async def atransform(self, feature_search: Dataset, problem: Problem, join_strategies: TablesWithJoinStrategies,
                         conn: DuckDBPyConnection, feature: Feature) -> list[Feature]:
        await asyncio.sleep(0)
        if hash(str(feature)) < 2**16:
            return []
        else:
            return [agentune.analyze.util.copy.replace(feature, name=f'{feature.name}, transformed')]

class ToySyncFeatureDescriber(SyncFeatureDescriber):
    @override
    def description(self, feature: Feature) -> str:
        return f'{feature.name} described'

class ToyAsyncFeatureDescriber(FeatureDescriber):
    @override
    async def adescription(self, feature: Feature) -> str:
        await asyncio.sleep(0)
        return f'{feature.name} described'
    

@frozen
class ToySyncFeatureSelector(SyncFeatureSelector):
    features: list[FeatureWithFullStats] = field(factory=list)

    @override
    def add_feature(self, feature_with_stats: FeatureWithFullStats) -> None:
        self.features.append(feature_with_stats)
    
    @staticmethod
    def some_metric(stats: FeatureStats) -> float:
        return float(len(stats.support))

    @override
    def select_final_features(self, problem: Problem) -> list[FeatureWithFullStats]:
        average_metric = math.nan if len(self.features) == 0 \
            else sum(ToySyncFeatureSelector.some_metric(x.stats.feature) for x in self.features) / len(self.features)
        selected = [x for x in self.features if ToySyncFeatureSelector.some_metric(x.stats.feature) >= average_metric]
        self.features.clear() # Prepare for reuse
        return selected

@frozen
class ToyAsyncFeatureSelector(FeatureSelector):
    features: list[FeatureWithFullStats] = field(factory=list)

    @override
    async def aadd_feature(self, feature_with_stats: FeatureWithFullStats) -> None:
        await asyncio.sleep(0)
        self.features.append(feature_with_stats)

    @override
    async def aselect_final_features(self, problem: Problem) -> list[FeatureWithFullStats]:
        await asyncio.sleep(0)
        average_metric = math.nan if len(self.features) == 0 \
            else sum(ToySyncFeatureSelector.some_metric(x.stats.feature) for x in self.features) / len(self.features)
        selected = [x for x in self.features if ToySyncFeatureSelector.some_metric(x.stats.feature) >= average_metric]
        self.features.clear() # Prepare for reuse
        return selected

@frozen(eq=False, hash=False)
class ToyAllFeatureSelector(SyncFeatureSelector):
    _added_features: list[FeatureWithFullStats] = field(factory=list, init=False)

    @override
    def add_feature(self, feature_with_stats: FeatureWithFullStats) -> None:
        self._added_features.append(feature_with_stats)

    @override
    def select_final_features(self, problem: Problem) -> Sequence[FeatureWithFullStats]:
        selected = tuple(self._added_features)
        self._added_features.clear() # Prepare for reuse
        return selected


class ToyAsyncEnrichedFeatureSelector(EnrichedFeatureSelector):
    def some_metric(self, feature: Feature, enriched: pl.Series) -> float:
        if isinstance(feature, NumericFeature):
            return cast(float, enriched.mean())
        elif isinstance(feature, CategoricalFeature):
            return enriched.n_unique()
        elif isinstance(feature, BoolFeature):
            return cast(int, enriched.sum())
        else:
            raise TypeError(f'Unknown feature type: {type(feature)}')

    @override
    async def aselect_features(self, features: Sequence[Feature],
                               enriched_data: DatasetSource, problem: Problem,
                               conn: DuckDBPyConnection) -> Sequence[Feature]:
        await asyncio.sleep(0)
        assert set(enriched_data.schema.names) == set([feature.name for feature in features] + [problem.target_column.name])

        dataset = enriched_data.to_dataset(conn)
        average_metric = math.nan if len(features) == 0 \
            else sum(self.some_metric(feature, dataset.data[feature.name])
                     for feature in features) / len(features)
        return [feature for feature in features
                if self.some_metric(feature, dataset.data[feature.name]) >= average_metric]

class ToySyncEnrichedFeatureSelector(SyncEnrichedFeatureSelector):
    def some_metric(self, feature: Feature, enriched: pl.Series) -> float:
        if isinstance(feature, NumericFeature):
            return cast(float, enriched.mean())
        elif isinstance(feature, CategoricalFeature):
            return enriched.n_unique()
        elif isinstance(feature, BoolFeature):
            return cast(int, enriched.sum())
        else:
            raise TypeError(f'Unknown feature type: {type(feature)}')

    @override
    def select_features(self, features: Sequence[Feature],
                        enriched_data: DatasetSource, problem: Problem,
                        conn: DuckDBPyConnection) -> Sequence[Feature]:
        assert set(enriched_data.schema.names) == set([feature.name for feature in features] + [problem.target_column.name])

        dataset = enriched_data.to_dataset(conn)
        average_metric = math.nan if len(features) == 0 \
            else sum(self.some_metric(feature, dataset.data[feature.name])
                     for feature in features) / len(features)
        return [feature for feature in features
                if self.some_metric(feature, dataset.data[feature.name]) >= average_metric]
