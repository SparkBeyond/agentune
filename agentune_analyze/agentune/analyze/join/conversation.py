from __future__ import annotations

import datetime
from typing import override

import attrs
import more_itertools
import polars as pl
from attrs import field, frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.core import types
from agentune.analyze.core.database import (
    ArtIndex,
    DuckdbIndex,
    DuckdbName,
    DuckdbTable,
)
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Field, dtype_is
from agentune.analyze.join.base import JoinStrategy
from agentune.analyze.util.duckdbutil import results_iter


@frozen
class Message:
    role: str
    timestamp: datetime.datetime
    content: str

@frozen
class Conversation:
    messages: tuple[Message, ...] = field()

    @messages.validator
    def _validate_messages(self, _attribute: attrs.Attribute, value: tuple[Message, ...]) -> None:
        if not more_itertools.is_sorted(value, key=lambda m: m.timestamp): # NOTE we permit messages with the same timestamp
            raise ValueError('Messages must be sorted by timestamp')

@frozen
class ConversationJoinStrategy[K](JoinStrategy):
    """A strategy for storing conversations in a secondary table with one row per message.

    It is joined to the main table by the conversation id column, whose type K can be any type that supports 
    indexing and equality comparisons.

    tparams:
        K: the type of the conversation id column (self.main_table_id_column in the main table, and self.id_column in the secondary table).
           This can be any type that supports duckdb indexing and equality comparisons in SQL queries.
    """
    name: str
    table: DuckdbTable
    main_table_id_column: Field
    id_column: Field = field()
    timestamp_column: Field = field(validator=dtype_is(types.timestamp))
    role_column: Field = field(validator=attrs.validators.or_(dtype_is(types.string), dtype_is(types.EnumDtype)))
    content_column: Field = field(validator=dtype_is(types.string))

    @id_column.validator
    def _validate_id_column(self, _attribute: attrs.Attribute, value: Field) -> None:
        if value.dtype != self.main_table_id_column.dtype:
            raise ValueError(f'ID column {value.name} has dtype {value.dtype}, '
                             f'but main table ID column {self.main_table_id_column.name} has dtype {self.main_table_id_column.dtype}')

    @staticmethod
    def on_table(name: str, table: DuckdbTable, main_table_id_column: str, id_column: str,
                 timestamp_column: str, role_column: str, content_column: str) -> ConversationJoinStrategy[K]:
        relevant_table = DuckdbTable(table.name, table.schema.select(id_column, timestamp_column, role_column, content_column))
        return ConversationJoinStrategy[K](
            name, relevant_table,
            Field(main_table_id_column, table.schema[id_column].dtype),
            table.schema[id_column],
            table.schema[timestamp_column],
            table.schema[role_column],
            table.schema[content_column]
        )

    @property
    @override
    def index(self) -> DuckdbIndex:
        return ArtIndex(
            name=DuckdbName(f'art_by_{self.id_column.name}', self.table.name.database, self.table.name.schema),
            cols=(self.id_column.name, )
        )

    def get_conversation(self, conn: DuckDBPyConnection, id: K) -> Conversation | None:
        """Get a single conversation, given the id (from the primary table)."""
        conn.execute(f'''SELECT "{self.timestamp_column.name}", "{self.role_column.name}", "{self.content_column.name}"
                        FROM {self.table.name}
                        WHERE "{self.id_column.name}" = $1
                        ORDER BY "{self.timestamp_column.name}"''', [id])
        results = tuple(Message(
            timestamp=row[0],
            role=row[1],
            content=row[2]
        ) for row in results_iter(conn))
        if not results:
            return None
        return Conversation(results)

    def get_conversations(self, ids: list[K] | pl.Series | pl.DataFrame | Dataset, conn: DuckDBPyConnection) -> tuple[Conversation | None, ...]:
        """Batch get conversations, joined to the main table's series of conversation IDs.
        
        Args:
            ids: a list or series of conversation IDs, or a dataframe or dataset with the column named by self.main_table_id_column
        """
        if isinstance(ids, Dataset):
            ids = ids.data[self.main_table_id_column.name]
        elif isinstance(ids, pl.DataFrame):
            ids = ids[self.main_table_id_column.name]
        
        if isinstance(ids, pl.Series):
            ids = ids.to_list()

        if len(ids) == 0:
            return ()
        
        conn.execute(f'''SELECT "{self.id_column.name}", "{self.timestamp_column.name}", "{self.role_column.name}", "{self.content_column.name}"
                         FROM {self.table.name}
                         WHERE "{self.id_column.name}" IN $1
                         ORDER BY "{self.id_column.name}", "{self.timestamp_column.name}"''', [ids])

        messages: list[Message] = []
        conversation_id = None
        conversations: dict[K, Conversation] = {}
        for (row_conversation_id, timestamp, role, content) in results_iter(conn):
            if row_conversation_id != conversation_id:
                if conversation_id is not None:
                    conversations[conversation_id] = Conversation(tuple(messages))
                messages = []
                conversation_id = row_conversation_id
            messages.append(Message(role, timestamp, content))
        if conversation_id is not None:
            conversations[conversation_id] = Conversation(tuple(messages))

        return tuple(conversations.get(id) for id in ids)
    
    def ids_exist(self, ids: list[K] | pl.Series | pl.DataFrame | Dataset, conn: DuckDBPyConnection) -> pl.Series:
        """Check which conversation IDs exist in the conversation table.
        
        Args:
            ids: a list or series of conversation IDs, or a dataframe or dataset with the column named by self.main_table_id_column

        Returns:
            A polars Series of booleans indicating whether each ID exists in the conversation table.
        """
        if isinstance(ids, Dataset):
            ids = ids.data[self.main_table_id_column.name]
        elif isinstance(ids, pl.DataFrame):
            ids = ids[self.main_table_id_column.name]

        if len(ids) == 0:
            return pl.Series([], dtype=pl.Boolean)

        # Register a dataframe with input IDs and explicit ordering
        input_df = pl.DataFrame({
            'id': ids
        })
        temp_table_name = f'{self.table.name.name}_ids_exist_temp'
        conn.register(temp_table_name, input_df)

        try:
            # Use EXISTS to check for existence
            output = conn.sql(f'''
                SELECT EXISTS(
                    SELECT * FROM {self.table.name}
                    WHERE "{self.id_column.name}" = input.id
                ) AS exists
                FROM {temp_table_name} input
            ''').pl()

            return output['exists']
        finally:
            # Clean up the temporary registered table
            conn.unregister(temp_table_name)
