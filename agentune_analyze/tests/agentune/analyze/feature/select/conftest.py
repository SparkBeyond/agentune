from collections.abc import Sequence

import polars as pl
import pytest

from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.dataset import Dataset, DatasetSource
from agentune.analyze.core.schema import Schema
from agentune.analyze.feature.base import BoolFeature, CategoricalFeature, FloatFeature
from agentune.analyze.join.base import JoinStrategy


class _Num(FloatFeature):
    def __init__(self, name: str):
        self.name = name
        self.description = f'Feature {name}'
        self.technical_description = ''
        self.default_for_missing = 0.0
        self.default_for_nan = 0.0
        self.default_for_infinity = 0.0
        self.default_for_neg_infinity = 0.0

    @property
    def code(self) -> str:
        return '<test>'

    @property
    def params(self) -> Schema:
        return Schema(())

    @property
    def secondary_tables(self) -> Sequence[DuckdbTable]:
        return []

    @property
    def join_strategies(self) -> Sequence[JoinStrategy]:
        return []


class _Cat(CategoricalFeature):
    _categories: tuple[str, ...]

    def __init__(self, name: str):
        self.name = name
        self.description = f'Categorical feature {name}'
        self.technical_description = ''
        self._categories = ()
        self.default_for_missing = ''

    @property
    def code(self) -> str:
        return '<test>'

    @property
    def params(self) -> Schema:
        return Schema(())

    @property
    def secondary_tables(self) -> Sequence[DuckdbTable]:
        return []

    @property
    def join_strategies(self) -> Sequence[JoinStrategy]:
        return []

    @property
    def categories(self) -> tuple[str, ...]:
        return self._categories

    @categories.setter
    def categories(self, value: tuple[str, ...]) -> None:
        self._categories = tuple(value)


class _Bool(BoolFeature):
    def __init__(self, name: str):
        self.name = name
        self.description = f'Boolean feature {name}'
        self.technical_description = ''
        self.default_for_missing = False

    @property
    def code(self) -> str:
        return '<test>'

    @property
    def params(self) -> Schema:
        return Schema(())

    @property
    def secondary_tables(self) -> Sequence[DuckdbTable]:
        return []

    @property
    def join_strategies(self) -> Sequence[JoinStrategy]:
        return []


class EnrichedBuilder:
    """Builder that produces Feature objects and a DatasetSource for the enriched API.

    Usage:
        builder = EnrichedBuilder()
        features, source = builder.build(df, target_col)
    """

    def build(self, df: pl.DataFrame, target_col: str) -> tuple[list[FloatFeature | CategoricalFeature | BoolFeature], DatasetSource]:
        feature_cols = [c for c in df.columns if c != target_col]
        features: list[FloatFeature | CategoricalFeature | BoolFeature] = []
        for c in feature_cols:
            dtype = df.schema[c]
            if dtype == pl.Utf8:
                features.append(_Cat(c))
            elif dtype == pl.Boolean:
                features.append(_Bool(c))
            else:
                features.append(_Num(c))
        dataset = Dataset.from_polars(df.select([*feature_cols, target_col]))
        source = dataset.as_source()
        return features, source


def pytest_configure(config: pytest.Config) -> None:
    # Register local markers for this test subtree to avoid PytestUnknownMarkWarning
    config.addinivalue_line('markers', 'slow: marks tests as slow (set RUN_SLOW=1 to include)')
