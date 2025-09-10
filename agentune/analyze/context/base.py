from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

from attrs import field, frozen
from frozendict import frozendict

from agentune.analyze.core.database import DuckdbIndex, DuckdbName, DuckdbTable
from agentune.analyze.util.attrutil import frozendict_converter


class ContextDefinition(ABC):
    """A way to use an indexed DB table as a particular type of context, e.g. a lookup table (defined by subclasses)."""

    @property
    @abstractmethod
    def table(self) -> DuckdbTable:
        """The table used and the schema of the columns used. This is often a subset of the columns originally in that table.

        This is known to be an incomplete API; some classes may use multiple tables, or particular columns from the main table. #191
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str: 
        """The context definition name; not the same as the backing table name.
        Context definition names are expected to be unique within some scope, e.g. a pipeline run,
        even when defining multiple contexts on the same table.
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
    context_definitions: frozendict[str, ContextDefinition] = field(converter=frozendict_converter)

    def __getitem__(self, name: str) -> ContextDefinition:
        return self.context_definitions[name]

    def __iter__(self) -> Iterator[ContextDefinition]:
        return iter(self.context_definitions.values())

    def __len__(self) -> int:
        return len(self.context_definitions)

    @staticmethod
    def from_list(context_definitions: Sequence[ContextDefinition]) -> TableWithContextDefinitions:
        tables = [c.table for c in context_definitions]
        if len(set(tables)) != 1:
            raise ValueError(f'Context definitions do not all refer to the same table: {set(tables)}')
        if len({c.name for c in context_definitions}) != len(context_definitions):
            raise ValueError('Context definitions have duplicate names')

        return TableWithContextDefinitions(
            tables[0],
            frozendict({c.name: c for c in context_definitions})
        )


@frozen
class TablesWithContextDefinitions:
    tables: frozendict[DuckdbName, TableWithContextDefinitions] = field(converter=frozendict_converter)

    @staticmethod
    def from_list(tables: Sequence[TableWithContextDefinitions]) -> TablesWithContextDefinitions:
        if len({t.table.name for t in tables}) != len(tables):
            raise ValueError('Tables have duplicate names')
        return TablesWithContextDefinitions(frozendict({
            t.table.name: t for t in tables
        }))

    @staticmethod
    def group(context_definitions: Sequence[ContextDefinition]) -> TablesWithContextDefinitions:
        return TablesWithContextDefinitions(frozendict({
            name: TableWithContextDefinitions.from_list(list(group))
            for name, group in itertools.groupby(context_definitions, lambda c: c.table.name)
        }))


    def __getitem__(self, name: DuckdbName) -> TableWithContextDefinitions:
        return self.tables[name]
    
    def __iter__(self) -> Iterator[TableWithContextDefinitions]:
        return iter(self.tables.values())
    
    def __len__(self) -> int:
        return len(self.tables)
    
    
