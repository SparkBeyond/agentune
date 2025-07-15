from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Literal, Self, override

import polars as pl
from attrs import field, frozen
from duckdb import DuckDBPyConnection, DuckDBPyRelation

from agentune.analyze.core.database import ArtIndex, DatabaseIndex, DatabaseTable
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Field


class ContextDefinition(ABC):
    """A way to use an indexed DB table as a particular type of context, e.g. a lookup table (defined by subclasses)."""

    @property
    @abstractmethod
    def table(self) -> DatabaseTable: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def index(self) -> DatabaseIndex: ...

    
type ContextBuilder[TDef: ContextDefinition, TContext: ContextObject] = Callable[[TDef], TContext]
'''Connects a context API to an implementation.
'''

class ContextObject(ABC):
    """A logical interface backed by an indexed table, with python methods representing the expected queries.
    Contains a live connection.
    """

    @property
    @abstractmethod
    def definition(self) -> ContextDefinition: ...
    
    @property
    @abstractmethod
    def conn(self) -> DuckDBPyConnection: ...

    def relation(self) -> DuckDBPyRelation:
        return self.conn.table(self.relation_name)
    
    @property
    def relation_name(self) -> str:
        return self.definition.table.name
    
    @property
    def name(self) -> str: 
        """The context object name; not the same as the backing relation name.
        Context object names are expected to be unique within some scope, e.g. a pipeline run,
        even when defining multiple context objects for the same table.
        """
        return self.definition.name

@frozen
class TableWithContextDefinitions:
    table: DatabaseTable
    context_definitions: tuple[ContextDefinition, ...]

@frozen
class TablesWithContextDefinitions:
    tables: Mapping[str, TableWithContextDefinitions]

    @classmethod
    def from_list(cls, tables: Iterable[TableWithContextDefinitions]) -> Self:
        return cls({t.table.name: t for t in tables})

    def __getitem__(self, name: str) -> TableWithContextDefinitions:
        return self.tables[name]
    
    def __iter__(self) -> Iterator[TableWithContextDefinitions]:
        return iter(self.tables.values())
    
    def __len__(self) -> int:
        return len(self.tables)
    

@frozen
class TableWithContexts:
    table: DatabaseTable
    contexts: tuple[ContextObject, ...]

@frozen
class TablesWithContexts:
    tables: Mapping[str, TableWithContexts]

    @classmethod
    def from_list(cls, tables: Iterable[TableWithContexts]) -> Self:
        return cls({t.table.name: t for t in tables})
    

@frozen
class LookupContextDefinition(ContextDefinition):
    name: str
    table: DatabaseTable
    key_col: Field
    value_cols: tuple[Field, ...] # can be used to restrict the available value columns from what's in the table

    @property
    @override
    def index(self) -> DatabaseIndex:
        return ArtIndex(
            name=f'art_by_{self.key_col.name}',
            cols=(self.key_col.name, )
        )

class LookupContext[K](ContextObject):
    @property
    @abstractmethod
    def definition(self) -> LookupContextDefinition: ...

    @abstractmethod
    def get(self, key: K, value_col: str) -> Any | None: 
        """Return the value for the given key, or None if not found."""
        ...

    @abstractmethod
    def get_many(self, key: K, value_cols: Iterable[str]) -> tuple | None: 
        """Return a list of values corresponding to the value_cols, or None if not found."""
        ...

    @abstractmethod
    def get_batch(self, keys: Iterable[K], value_cols: Iterable[str]) -> Dataset: 
        """Return a dataset with one row per key, in the order of the input keys. 
        
        If a key is not found, the matching output row will contain nulls for all value columns
        (this cannot be distinguished from a row where the key is found but all value columns are null).
        The output dataset will contain the key column as well as the selected value columns.
        """
        ...

@frozen
class KtsContextDefinition(ContextDefinition):
    name: str
    table: DatabaseTable
    key_col: Field
    date_col: Field # Should be of type timestamp
    value_cols: Iterable[Field] # can be used to restrict the available value columns from what's in the table

    @property
    @override
    def index(self) -> DatabaseIndex:
        return ArtIndex(
            name=f'art_by_{self.key_col.name}',
            cols=(self.key_col.name, )
        )

@frozen
class TimeWindow:
    start: datetime.datetime
    end: datetime.datetime
    include_start: bool
    include_end: bool
    sample_maxsize: int | None = None

@frozen
class TimeSeries:
    """A single (unkeyed, potentially sliced) TimeSeries is represented as a dataframe with a datetime column and one or more value columns."""

    dataset: Dataset
    date_col_name: str
    date_col: Field = field(init=False)
    value_cols: Iterable[Field] = field(init=False)

    @date_col.default
    def _date_col_default(self) -> Field:
        return self.dataset.schema[self.date_col_name]
    
    @value_cols.default
    def _value_cols_default(self) -> list[Field]:
        return [f for f in self.dataset.schema.cols if f.name != self.date_col_name]
    
    def slice(self, window: TimeWindow) -> TimeSeries:
        """WARNING: the downsampling logic does not produce the same result as slicing inside duckdb.
        
        This is mostly useful for tests, not for implementing features, since unsliced and unsampled time series
        shouldn't normally appear as values in memory.
        """
        closed: Literal['left', 'right', 'both', 'none'] = \
            'both' if window.include_start and window.include_end else \
            'left' if window.include_start else \
            'right' if window.include_end else \
            'none'
        new_df: pl.DataFrame = self.dataset.data.filter(
            pl.col(self.date_col_name).is_between(window.start, window.end, closed=closed)
        )
        if window.sample_maxsize is not None:
            new_df = new_df.sample(n=window.sample_maxsize, seed=42)
        return TimeSeries(Dataset(self.dataset.schema, new_df), self.date_col_name)
        

class KtsContext[K](ContextObject):
    @property
    @abstractmethod
    def definition(self) -> KtsContextDefinition: ...

    @abstractmethod
    def get(self, key: K, window: TimeWindow, value_cols: Iterable[str]) -> TimeSeries | None: ...

    # I wanted to implemented a get_batch, taking `keys: Iterable[K]` and returning `Iterable[Option[TimeSeries]]`,
    # but I don't know how to implement the downsampling in way that would be more efficient than calling it once per key.
    
