from __future__ import annotations

from collections.abc import Sequence
from typing import Any, override

from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.context.base import LookupContext, LookupContextDefinition
from agentune.analyze.core.dataset import Dataset, duckdb_to_dataset


@frozen
class LookupContextImpl[K](LookupContext):
    conn: DuckDBPyConnection
    definition: LookupContextDefinition

    @override
    def get(self, key: K, value_col: str) -> Any | None:
        # TODO relation and column name escaping (double any quotes inside them)
        results = self.conn.sql(f'select "{value_col}" from "{self.relation_name}" where "{self.definition.key_col.name}" = ?', 
                                params=[key]).fetchall()
        assert (len(results) <= 1)
        return results[0][0] if results else None

    @override
    def get_many(self, key: K, value_cols: Sequence[str]) -> tuple | None:
        value_cols_query = ', '.join(f'"{col}"' for col in value_cols)
        results = self.conn.sql(f'select {value_cols_query} from "{self.relation_name}" where "{self.definition.key_col.name}" = ?',
                                params=[key]).fetchall()
        assert (len(results) <= 1)
        return results[0] if results else None

    @override
    def get_batch(self, keys: Sequence[K], value_cols: Sequence[str]) -> Dataset:
        value_cols_query = ', '.join(f't."{col}"' for col in value_cols)
        relation = self.conn.sql(f'''WITH key_list AS (
                                        SELECT 
                                            value AS key_value,
                                            row_number() OVER () AS position
                                        FROM unnest(?) AS t(value)
                                    )
                                    SELECT k.key_value "{self.definition.key_col.name}", {value_cols_query} 
                                    FROM key_list k
                                    LEFT JOIN {self.relation_name} t ON k.key_value = t."{self.definition.key_col.name}"
                                    ORDER BY k.position
                                    ''', params=[tuple(keys)]) 
        return duckdb_to_dataset(relation)

