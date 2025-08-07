from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Self

from attrs import field, frozen
from frozendict import frozendict

from agentune.analyze.core.database import DuckdbIndex, DuckdbTable
from agentune.analyze.util.attrutil import frozendict_converter


# TODO a better name for this 
class ContextDefinition(ABC):
    """A way to use an indexed DB table as a particular type of context, e.g. a lookup table (defined by subclasses)."""

    @property
    @abstractmethod
    def table(self) -> DuckdbTable: ...

    @property
    @abstractmethod
    def name(self) -> str: 
        """The context object name; not the same as the backing relation name.
        Context object names are expected to be unique within some scope, e.g. a pipeline run,
        even when defining multiple context objects for the same table.
        """
        ...

    @property
    @abstractmethod
    def index(self) -> DuckdbIndex: 
        """The index that needs to be created on the table to support efficient access."""
        ...
    

@frozen
class TableWithContextDefinitions:
    table: DuckdbTable
    context_definitions: tuple[ContextDefinition, ...]

@frozen
class TablesWithContextDefinitions:
    tables: frozendict[str, TableWithContextDefinitions] = field(converter=frozendict_converter)

    @classmethod
    def from_list(cls, tables: Sequence[TableWithContextDefinitions]) -> Self:
        return cls({t.table.name: t for t in tables})

    def __getitem__(self, name: str) -> TableWithContextDefinitions:
        return self.tables[name]
    
    def __iter__(self) -> Iterator[TableWithContextDefinitions]:
        return iter(self.tables.values())
    
    def __len__(self) -> int:
        return len(self.tables)
    
    
