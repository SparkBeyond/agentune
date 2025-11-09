from __future__ import annotations

import logging
from functools import cached_property

import attrs
from attrs import field, frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbName, DuckdbTable
from agentune.analyze.core.dataset import DatasetSource
from agentune.analyze.core.duckdbio import DatasetSourceFromDuckdb
from agentune.analyze.core.schema import Schema
from agentune.analyze.util.duckdbutil import transaction_scope

_logger = logging.getLogger(__name__)

@frozen
class SplitDuckdbTable:
    """A table with extra columns that split the data into several nonexclusive categories.

    1. Every row is either train or test.
    2. Every train row can be part of the feature_search and/or the feature_eval datasets,
       independently of each other.

    Implementation-wise, there are several bit columns: is_train, is_feature_search, is_feature_eval.
    """
    table: DuckdbTable = field()
    is_train_col_name: str = '_is_train'
    is_feature_search_col_name: str = '_is_feature_search'
    is_feature_eval_col_name: str = '_is_feature_eval'

    @table.validator
    def _table_validator(self, _attribute: attrs.Attribute, table: DuckdbTable) -> None:
        for col in (self.is_train_col_name, self.is_feature_search_col_name, self.is_feature_eval_col_name):
            if col not in table.schema.names:
                raise ValueError(f'Marker column {col} not found in table {table.name}')
            if table.schema[col].dtype != types.boolean:
                raise ValueError(f'Marker column {col} has dtype {table.schema[col].dtype}, expected boolean')

    @property
    def schema(self) -> Schema:
        return self.table.schema

    @property
    def schema_without_split_columns(self) -> Schema:
        return self.table.schema.drop(self.is_train_col_name, self.is_feature_search_col_name, self.is_feature_eval_col_name)

    @cached_property
    def _select_orig_col_names(self) -> str:
        return ', '.join(f'"{name}"' for name in self.schema_without_split_columns.names)

    def train(self) -> DatasetSource:
        return DatasetSourceFromDuckdb(self.schema_without_split_columns,
                                       lambda conn: conn.sql(f'SELECT {self._select_orig_col_names} FROM {self.table.name} WHERE "{self.is_train_col_name}"'),
                                       allow_cheap_size=True)

    def test(self) -> DatasetSource:
        return DatasetSourceFromDuckdb(self.schema_without_split_columns,
                                       lambda conn: conn.sql(f'SELECT {self._select_orig_col_names} FROM {self.table.name} WHERE NOT "{self.is_train_col_name}"'),
                                       allow_cheap_size=True)

    def feature_search(self) -> DatasetSource:
        return DatasetSourceFromDuckdb(self.schema_without_split_columns,
                                       lambda conn: conn.sql(f'SELECT {self._select_orig_col_names} FROM {self.table.name} WHERE "{self.is_feature_search_col_name}"'),
                                       allow_cheap_size=True)

    def feature_eval(self) -> DatasetSource:
        return DatasetSourceFromDuckdb(self.schema_without_split_columns,
                                       lambda conn: conn.sql(f'SELECT {self._select_orig_col_names} FROM {self.table.name} WHERE "{self.is_feature_eval_col_name}"'),
                                       allow_cheap_size=True)

    def drop_split_columns(self, conn: DuckDBPyConnection) -> None:
        with transaction_scope(conn):
            conn.execute(f'''ALTER TABLE {self.table.name} DROP COLUMN "{self.is_train_col_name}";''')
            conn.execute(f'''ALTER TABLE {self.table.name} DROP COLUMN "{self.is_feature_search_col_name}";''')
            conn.execute(f'''ALTER TABLE {self.table.name} DROP COLUMN "{self.is_feature_eval_col_name}";''')


def _add_train_sample(conn: DuckDBPyConnection, table_name: DuckdbName, is_train_col_name: str,
                      new_col_name: str, count: int) -> None:
    """Intermediate step: add a new column which is true for a random sample of up to `count` train rows."""
    conn.execute(f'''ALTER TABLE {table_name}
                     ADD COLUMN "{new_col_name}" BOOLEAN
                     DEFAULT false''')
    conn.execute(f'ALTER TABLE {table_name} ALTER COLUMN "{new_col_name}" SET NOT NULL')
    # 'USING SAMPLE RESERVOIR(count ROWS)' is not stable because duckdb is multithreaded,
    # so we use hash(rowid).
    conn.execute(f'''UPDATE {table_name}
                     SET "{new_col_name}" = true
                     WHERE "{is_train_col_name}" AND rowid IN (
                        SELECT rowid FROM {table_name}
                        WHERE "{is_train_col_name}"
                        ORDER BY hash(rowid), rowid -- in case of hash collisions
                        LIMIT $1
                     )''', [count])
    conn.execute(f'ALTER TABLE {table_name} ALTER COLUMN "{new_col_name}" DROP DEFAULT')

default_feature_search_size = 1000
default_feature_eval_size = 1000

def split_duckdb_table(conn: DuckDBPyConnection, table_name: DuckdbName | str,
                       train_fraction: float = 0.8,
                       feature_search_size: int = default_feature_search_size,
                       feature_eval_size: int = default_feature_eval_size,
                       is_train_col_name: str = '_is_train',
                       is_feature_search_col_name: str = '_is_feature_search',
                       is_feature_eval_col_name: str = '_is_feature_eval') -> SplitDuckdbTable:
    """Add split columns to an existing table, so that:
    1. Mark train_fraction of the rows as train (the rest are test)
    2. Out of train, mark feature_search_size rows as feature_search (this is an absolute cap, not a fraction)
    3. Out of train, and independently of feature_search, mark feature_eval_size rows as feature_eval
    """
    if train_fraction < 0 or train_fraction > 1:
        raise ValueError(f'train_fraction must be between 0 and 1, got {train_fraction}')

    if isinstance(table_name, str):
        table_name = DuckdbName.qualify(table_name, conn)

    # Ideally this would be a single transaction, but data updates and schema changes can't be interleaved in one transaction.
    # So we need at least three transactions.

    try:
        with transaction_scope(conn):
            # We can't use `when random() < train_fraction` because random() isn't deterministic since duckdb is
            # multithreaded (not even if we call setseed() first). And we can't set threads = 1, that would impact
            # code running on other (python) threads. So we use 'hash(rowid)' instead.
            # See limitations at: https://duckdb.org/docs/stable/sql/statements/select.html#row-ids
            # Tables that had undergone deletions and insertions won't have the same rowids as tables with the same data
            # inserted in one go, but SQL has no notion of row order, so this is already much better than another database
            # would give us.
            conn.execute(f'''ALTER TABLE {table_name}
                             ADD COLUMN "{is_train_col_name}" BOOLEAN
                             DEFAULT false''') # Accessing rowid or other columns not allowed in DEFAULT clause, so we can't do this in one go
            conn.execute(f'ALTER TABLE {table_name} ALTER COLUMN "{is_train_col_name}" DROP DEFAULT')
            conn.execute(f'ALTER TABLE {table_name} ALTER COLUMN "{is_train_col_name}" SET NOT NULL')
            conn.execute(f'''UPDATE {table_name}
                             SET "{is_train_col_name}" = true
                             WHERE hash(rowid) % 1000 < ?''', [int(train_fraction * 1000)])

        with transaction_scope(conn):
            _add_train_sample(conn, table_name, is_train_col_name, is_feature_search_col_name, feature_search_size)

        with transaction_scope(conn):
            _add_train_sample(conn, table_name, is_train_col_name, is_feature_eval_col_name, feature_eval_size)
    except Exception as e:
        try:
            conn.execute(f'ALTER TABLE {table_name} DROP COLUMN IF EXISTS "{is_train_col_name}"')
            conn.execute(f'ALTER TABLE {table_name} DROP COLUMN IF EXISTS "{is_feature_search_col_name}"')
            conn.execute(f'ALTER TABLE {table_name} DROP COLUMN IF EXISTS "{is_feature_eval_col_name}"')
        except Exception as e2:
            raise e2 from e
        raise

    return SplitDuckdbTable(DuckdbTable.from_duckdb(table_name, conn),
                            is_train_col_name, is_feature_search_col_name, is_feature_eval_col_name)
