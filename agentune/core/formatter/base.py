"""Base classes for data formatting.

This module defines the core interfaces for data formatting operations.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import attrs
import polars as pl
from duckdb import DuckDBPyConnection

from agentune.analyze.join.base import JoinStrategy
from agentune.core.database import DuckdbTable
from agentune.core.dataset import Dataset
from agentune.core.schema import Schema
from agentune.core.util.cattrutil import UseTypeTag


@attrs.define
class DataFormatter(ABC, UseTypeTag):
    """Abstract base class for data formatting strategies.
    
    Similar to Feature, this defines what data the formatter needs and provides
    a method to format batches of data into string representations.
    """
    
    name: str

    @property
    @abstractmethod
    def description(self) -> str | None:
        """Description of the results produced by the formatter."""
        ...

    @property
    @abstractmethod
    def params(self) -> Schema:
        """Columns of the main table used by the formatter."""
        ...
    
    @property
    @abstractmethod
    def secondary_tables(self) -> Sequence[DuckdbTable]:
        """Secondary tables used by the formatter (via SQL queries)."""
        ...

    @property
    @abstractmethod
    def join_strategies(self) -> Sequence[JoinStrategy]:
        """Join strategies used by the feature (via python methods on the context definitions)."""
        ...

    @abstractmethod
    async def aformat_batch(self, input: Dataset, conn: DuckDBPyConnection) -> pl.Series:
        """Format a batch of data into string representations.
        
        Args:
            input: Dataset containing the data to format
            conn: Database connection for accessing secondary tables
            
        Returns:
            pl.Series of strings with name=self.name
        """
        ...


