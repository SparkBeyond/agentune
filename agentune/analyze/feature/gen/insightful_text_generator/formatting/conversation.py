

from collections.abc import Sequence
from typing import override

import attrs
import polars as pl
from duckdb import DuckDBPyConnection

from agentune.analyze.feature.gen.insightful_text_generator.formatting.base import DataFormatter
from agentune.analyze.join.base import JoinStrategy
from agentune.analyze.join.conversation import ConversationJoinStrategy
from agentune.core.database import DuckdbTable
from agentune.core.dataset import Dataset
from agentune.core.schema import Field, Schema


@attrs.frozen
class ShortDateConversationFormatter(DataFormatter):
    """Formatter for conversation data with specific column structure.
    
    Groups by conversation_id, sorts by timestamp, and formats as:
    '--- Conversation {index + 1}/ ID {conversation_id} ---' (optional, based on flags)
    'Date: {date}'
    '[{time}] [{role}] {message}'
    ...
    '[{time}] [{role}] {message}'
    '[{field_name}] {field_value}' for each field in self.params. (optional)

    Args:
        conversation_strategy: ConversationJoinStrategy
        params_to_print: additional fields from main table to include in formatting
        include_conversation_id: whether to include conversation ID in the formatting
        include_in_batch_id: whether to include the number of each conversation in the batch ID

    """
    conversation_strategy: ConversationJoinStrategy
    params_to_print: tuple[Field, ...] = attrs.field(factory=tuple)  # additional fields from main table to include in formatting
    include_conversation_id: bool = False  # whether to include conversation ID in the formatting
    include_in_batch_id: bool = False  # whether to include the number of each conversation in the batch ID

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
        if self.include_in_batch_id and self.include_conversation_id:
            raise ValueError('Cannot include both conversation ID and batch ID in the formatting.')
        
        df = input.data

        conversations = self.conversation_strategy.get_conversations(conn=conn, ids=input)
        if len(conversations) != len(df):
            raise ValueError('Number of conversations does not match number of rows in input data')
        
        # Format each conversation into a string
        formatted_conversations = []
        # filter the dataframe to only include id column and params_to_print columns
        filtered_df = df.select([self.conversation_strategy.main_table_id_column.name, *[field.name for field in self.params_to_print]])
        for index, (row, conversation) in enumerate(zip(filtered_df.iter_rows(), conversations, strict=False)):
            if conversation is None:
                raise ValueError(f'Conversation missing for id: {row[0]}')

            text = []
            if self.include_conversation_id:
                text.append(f'--- Conversation ID: {row[0]} ---\n')
            if self.include_in_batch_id:
                text.append(f'--- Conversation {index + 1} ---\n')

            current_date = None
            for message in conversation.messages:
                date_str = message.timestamp.strftime('%Y-%m-%d')
                time_str = message.timestamp.strftime('%H:%M:%S')
                if date_str != current_date:
                    current_date = date_str
                    text.append(f'Date: {current_date}')
                text.append(f'[{time_str}] [{message.role}] {message.content}')
            
            # add information from the main table - filter the row for this conversation_id
            if self.params_to_print:
                # Map field names to their values in the row, which starts from index 1
                extra_fields = [f'[{field.name}] {row[i + 1]}' for i, field in enumerate(self.params_to_print)]
                text.extend(extra_fields)
            conversation_text = '\n'.join(text) + '\n'
            formatted_conversations.append(conversation_text)
        
        return pl.Series(name=self.name, values=formatted_conversations)
