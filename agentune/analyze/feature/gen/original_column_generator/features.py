from collections.abc import Sequence
from typing import override

import polars as pl
from attrs import field, frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.feature.base import (
    SyncBoolFeature,
    SyncCategoricalFeature,
    SyncFeature,
    SyncFloatFeature,
    SyncIntFeature,
)
from agentune.analyze.join.base import JoinStrategy
from agentune.core.database import DuckdbTable
from agentune.core.dataset import Dataset
from agentune.core.schema import Field, Schema


@frozen
class OriginalColumnFeature[T](SyncFeature[T]):
    """Generic feature that passes through an original column.
    
    This is a simple wrapper that extracts a column from the input dataset
    without any transformation.
    """
    input: Field

    @property
    @override
    def params(self) -> Schema:
        return Schema((self.input,))

    @property
    @override
    def secondary_tables(self) -> Sequence[DuckdbTable]:
        return ()

    @property
    @override
    def join_strategies(self) -> Sequence[JoinStrategy]:
        return ()

    @override
    def compute_batch(self, input: Dataset, conn: DuckDBPyConnection) -> pl.Series:
        return input.data.get_column(self.input.name)


@frozen
class OriginalIntFeature(OriginalColumnFeature[int], SyncIntFeature):
    """Integer feature that passes through an original column."""


@frozen
class OriginalFloatFeature(OriginalColumnFeature[float], SyncFloatFeature):
    """Float feature that passes through an original column."""
    # Redeclare with concrete types/defaults as required by FloatFeature
    default_for_missing: float = field(default=0.0)
    default_for_nan: float = field(default=0.0)
    default_for_infinity: float = field(default=0.0)
    default_for_neg_infinity: float = field(default=0.0)


@frozen
class OriginalBoolFeature(OriginalColumnFeature[bool], SyncBoolFeature):
    """Boolean feature that passes through an original column."""


@frozen
class OriginalCategoricalFeature(OriginalColumnFeature[str], SyncCategoricalFeature):
    """Categorical feature that passes through an original column."""
