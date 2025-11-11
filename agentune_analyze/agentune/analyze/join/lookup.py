from __future__ import annotations

from collections.abc import Sequence
from typing import Any, override

from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.core.database import (
    ArtIndex,
    DuckdbIndex,
    DuckdbName,
    DuckdbTable,
)
from agentune.analyze.core.dataset import Dataset, duckdb_to_dataset
from agentune.analyze.core.schema import Field
from agentune.analyze.join.base import JoinStrategy


@frozen
class LookupJoinStrategy[K](JoinStrategy):
    name: str
    table: DuckdbTable
    key_col: Field
    value_cols: tuple[Field, ...] # can be used to restrict the available value columns from what's in the table

    @staticmethod
    def on_table(name: str, table: DuckdbTable, key_col: str, *value_cols: str) -> LookupJoinStrategy[K]:
        relevant_table = DuckdbTable(table.name, table.schema.select(*(key_col, *value_cols)))
        return LookupJoinStrategy[K](
            name, relevant_table,
            table.schema[key_col],
            tuple(table.schema[col] for col in value_cols)
        )

    @property
    @override
    def index(self) -> DuckdbIndex:
        return ArtIndex(
            name=DuckdbName(f'art_by_{self.key_col.name}', self.table.name.database, self.table.name.schema),
            cols=(self.key_col.name, )
        )

    def get(self, conn: DuckDBPyConnection, key: K, value_col: str) -> Any | None:
        """Return the value for the given key, or None if not found."""
        results = conn.sql(f'select "{value_col}" from {self.table.name} where "{self.key_col.name}" = ?',
                           params=[key]).fetchall()
        match results:
            case []: return None
            case [(result, )]: return result
            case other: raise ValueError(f'Unexpected query result {other}')

    def get_many(self, conn: DuckDBPyConnection, key: K, value_cols: Sequence[str]) -> tuple | None:
        """Return a list of values corresponding to the value_cols, or None if not found."""
        value_cols_query = ', '.join(f'"{col}"' for col in value_cols)
        results = conn.sql(f'select {value_cols_query} from {self.table.name} where "{self.key_col.name}" = ?',
                           params=[key]).fetchall()
        match results:
            case []: return None
            case [tuple(result)]: return result
            case other: raise ValueError(f'Unexpected query result {other}')

    def get_batch(self, conn: DuckDBPyConnection, keys: Sequence[K], value_cols: Sequence[str]) -> Dataset:
        """Return a dataset with one row per key, in the order of the input keys. 
        
        If a key is not found, the matching output row will contain nulls for all value columns
        (this cannot be distinguished from a row where the key is found but all value columns are null).
        The output dataset will contain the key column as well as the selected value columns.
        """
        value_cols_query = ', '.join(f't."{col}"' for col in value_cols)
        relation = conn.sql(f'''WITH key_list AS (
                                SELECT 
                                    value AS key_value,
                                    row_number() OVER () AS position
                                FROM unnest(?) AS t(value)
                            )
                            SELECT k.key_value "{self.key_col.name}", {value_cols_query} 
                            FROM key_list k
                            LEFT JOIN {self.table.name} t ON k.key_value = t."{self.key_col.name}"
                            ORDER BY k.position
                            ''', params=[tuple(keys)]) 
        return duckdb_to_dataset(relation)

