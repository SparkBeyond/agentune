"""Base classes for data formatting.

This module defines the core interfaces for data formatting operations.
"""

from abc import ABC, abstractmethod
from typing import override

import attrs
import polars as pl
from duckdb import DuckDBPyConnection

from agentune.analyze.context.conversation import ConversationContext
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Schema


@attrs.define
class DataFormatter(ABC):
    """Abstract base class for data formatting strategies.
    
    Similar to Feature, this defines what data the formatter needs and provides
    a method to format batches of data into string representations.
    """
    
    name: str
    
    @abstractmethod
    async def aformat_batch(self, input: Dataset, conn: DuckDBPyConnection) -> pl.Series:
        """Format a batch of data into string representations.
        
        Args:
            input: Dataset containing the data to format
            conn: Database connection for accessing context tables
            
        Returns:
            pl.Series of strings with name=self.name
        """
        ...


@attrs.define
class ConversationFormatter(DataFormatter):
    """Formatter for conversation data with specific column structure.
    
    Groups by conversation_id, sorts by timestamp, and formats as:
    '[{timestamp}] [{role}] {message}'
    Also includes additional fields from the main table.
    '[{field_name}] {field_value}' for each field in self.params.
    """
    conversation_context: ConversationContext
    params: Schema
    
    @override
    async def aformat_batch(self, input: Dataset, conn: DuckDBPyConnection) -> pl.Series:
        """Format conversation data grouped by conversation_id."""
        df = input.data

        # get conversations from context
        conversations = self.conversation_context.get_conversations(conn=conn, ids=df[self.conversation_context.main_table_id_column.name])
        assert len(conversations) == len(df), 'Number of conversations does not match number of rows in input data'
        
        # Format each conversation into a string
        formatted_conversations = []
        # filter the dataframe to only include self.params columns
        filtered_df = df.select([self.conversation_context.main_table_id_column.name, *self.params.names])
        for row, conversation in zip(filtered_df.iter_rows(), conversations, strict=False):
            assert conversation is not None, f'Conversation missing for id: {row[0]}'
            # Format each message and join into conversation text
            text = [f'[{message.timestamp}] [{message.role}] {message.content}' for message in conversation.messages]
            # add information from the main table - filter the row for this conversation_id
            if self.params:
                extra_fields = [f'[{field_name}] {value}' for field_name, value in zip(self.params.names, row[1:], strict=False)]
                text.extend(extra_fields)
            conversation_text = '\n'.join(text)
            formatted_conversations.append(conversation_text)
        
        return pl.Series(name=self.name, values=formatted_conversations)
