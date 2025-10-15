"""Base classes for data formatting.

This module defines the core interfaces for data formatting operations.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import override

import attrs
import polars as pl
from duckdb import DuckDBPyConnection

from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.join.base import JoinStrategy
from agentune.analyze.join.conversation import ConversationJoinStrategy
from agentune.analyze.util.cattrutil import UseTypeTag


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


@attrs.frozen
class ConversationFormatter(DataFormatter):
    """Formatter for conversation data with specific column structure.
    
    Groups by conversation_id, sorts by timestamp, and formats as:
    '[{timestamp}] [{role}] {message}'
    Also includes additional fields from the main table.
    '[{field_name}] {field_value}' for each field in self.params.
    """
    conversation_strategy: ConversationJoinStrategy
    params_to_print: tuple[Field, ...] = attrs.field(factory=tuple)  # additional fields from main table to include in formatting

    @property
    def description(self) -> str | None:
        """Description of the formatter."""
        description = f'Full conversation transcription for {self.conversation_strategy.name}'
        if self.params_to_print:
            description += f", including fields: {', '.join(field.name for field in self.params_to_print)}"
        return description

    @property
    def params(self) -> Schema:
        cols = (self.conversation_strategy.main_table_id_column, *self.params_to_print)
        return Schema(cols=cols)

    @property
    def secondary_tables(self) -> Sequence[DuckdbTable]:
        return [self.conversation_strategy.table]
    
    @property
    def join_strategies(self) -> Sequence[JoinStrategy]:
        return [self.conversation_strategy]

    @override
    async def aformat_batch(self, input: Dataset, conn: DuckDBPyConnection) -> pl.Series:
        """Format conversation data grouped by conversation_id."""
        df = input.data

        conversations = self.conversation_strategy.get_conversations(conn=conn, ids=input)
        if len(conversations) != len(df):
            raise ValueError('Number of conversations does not match number of rows in input data')
        
        # Format each conversation into a string
        formatted_conversations = []
        # filter the dataframe to only include id column and params_to_print columns
        filtered_df = df.select([self.conversation_strategy.main_table_id_column.name, *[field.name for field in self.params_to_print]])
        for row, conversation in zip(filtered_df.iter_rows(), conversations, strict=False):
            if conversation is None:
                raise ValueError(f'Conversation missing for id: {row[0]}')
            # Format each message and join into conversation text
            text = [f'[{message.timestamp}] [{message.role}] {message.content}' for message in conversation.messages]
            # add information from the main table - filter the row for this conversation_id
            if self.params_to_print:
                # Map field names to their values in the row, which starts from index 1
                extra_fields = [f'[{field.name}] {row[i + 1]}' for i, field in enumerate(self.params_to_print)]
                text.extend(extra_fields)
            conversation_text = '\n'.join(text)
            formatted_conversations.append(conversation_text)
        
        return pl.Series(name=self.name, values=formatted_conversations)
