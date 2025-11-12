from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

from attrs import field, frozen
from frozendict import frozendict

from agentune.analyze.core.database import DuckdbIndex, DuckdbName, DuckdbTable
from agentune.analyze.util.attrutil import frozendict_converter
from agentune.analyze.util.cattrutil import UseTypeTag


class JoinStrategy(ABC, UseTypeTag):
    """A way to query data from particular DB tables, e.g. joining a secondary table using a particular column.

    Specific strategies are defined by subclasses.
    """

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
        """The join strategy name; not the same as the backing table name.
        Strategy names are expected to be unique within some scope, e.g. an analyze run,
        even when defining multiple strategies on the same table.
        """
        ...

    @property
    @abstractmethod
    def index(self) -> DuckdbIndex: 
        """An index that can be created on `self.table` to support efficient queries."""
        ...
    

@frozen
class TableWithJoinStrategies:
    table: DuckdbTable
    join_strategies: frozendict[str, JoinStrategy] = field(converter=frozendict_converter)

    def __getitem__(self, name: str) -> JoinStrategy:
        return self.join_strategies[name]

    def __iter__(self) -> Iterator[JoinStrategy]:
        return iter(self.join_strategies.values())

    def __len__(self) -> int:
        return len(self.join_strategies)

    @staticmethod
    def from_list(join_strategies: Sequence[JoinStrategy]) -> TableWithJoinStrategies:
        tables = [c.table for c in join_strategies]
        if len(set(tables)) != 1:
            raise ValueError(f'Join strategies do not all refer to the same table: {set(tables)}')
        if len({c.name for c in join_strategies}) != len(join_strategies):
            raise ValueError('Join strategies have duplicate names')

        return TableWithJoinStrategies(
            tables[0],
            frozendict({c.name: c for c in join_strategies})
        )


@frozen
class TablesWithJoinStrategies:
    tables: frozendict[DuckdbName, TableWithJoinStrategies] = field(converter=frozendict_converter)

    @staticmethod
    def from_list(tables: Sequence[TableWithJoinStrategies]) -> TablesWithJoinStrategies:
        if len({t.table.name for t in tables}) != len(tables):
            raise ValueError('Tables have duplicate names')
        return TablesWithJoinStrategies(frozendict({
            t.table.name: t for t in tables
        }))

    @staticmethod
    def group(join_strategies: Sequence[JoinStrategy]) -> TablesWithJoinStrategies:
        return TablesWithJoinStrategies(frozendict({
            name: TableWithJoinStrategies.from_list(list(group))
            for name, group in itertools.groupby(join_strategies, lambda c: c.table.name)
        }))

    @staticmethod
    def unflatten(tables: Sequence[DuckdbTable], join_strategies: Sequence[JoinStrategy]) -> TablesWithJoinStrategies:
        strategies_by_table_name = { k: list(v) for k, v in itertools.groupby(join_strategies, lambda c: c.table.name) }
        tables_with_strategies = [TableWithJoinStrategies(table, { strat.name: strat for strat in strategies_by_table_name.get(table.name, [])}) for table in tables]
        return TablesWithJoinStrategies.from_list(tables_with_strategies)


    def __getitem__(self, name: DuckdbName) -> TableWithJoinStrategies:
        return self.tables[name]
    
    def __iter__(self) -> Iterator[TableWithJoinStrategies]:
        return iter(self.tables.values())
    
    def __len__(self) -> int:
        return len(self.tables)
    
    
