"""Schema formatting for LLM prompts."""

from typing import override

import attrs
import polars as pl
from duckdb import DuckDBPyConnection

from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.core.dataset import Dataset
from agentune.core.formatter.base import TableFormatter, TablesFormatter
from agentune.core.sampler.base import DataSampler, RandomSampler, TableSampler
from agentune.core.sampler.table_samples import HeadTableSampler
from agentune.core.schema import Schema


@attrs.frozen
class MarkdownTableFormatter(TableFormatter):
    """Markdown table formatter.
    
    Formats a single table with its schema and sample data using markdown headers.
    The schema is displayed as a bulleted list of columns with their DuckDB types,
    and the sample data is formatted as CSV for readability.
    
    Args:
        markdown_level: The markdown header level to use for sections (default: 3).
        max_str: Maximum string length for cell values. Longer values are truncated with '...' (default: 100).
    
    Example:
        ### Schema:
        - id: INTEGER
        - name: VARCHAR
        - status: ENUM('active', 'inactive')
        - score: DOUBLE

        ### Sample Data:
        id,name,status,score
        1,Alice,active,95.5
        2,Bob,inactive,82.3
    """
    markdown_level: int = 3
    max_str: int = 100

    def _format_schema(self, schema: Schema) -> str:
        """Format schema to human-readable string."""
        lines = []
        for field in schema.cols:
            # Convert Dtype to simple string representation using duckdb_type
            dtype_str = repr(field.dtype.duckdb_type)
            lines.append(f'- {field.name}: {dtype_str}')

        return '\n'.join(lines)

    def _format_sample_data(self, dataset: Dataset) -> str:
        """Format sample data rows as table using CSV format."""
        # Only truncate string columns using Polars
        select_exprs = []
        for field in dataset.schema.cols:
            col_name = field.name
            # Check if column is a string type
            if field.dtype.polars_type in (pl.String, pl.Utf8):
                # Truncate long strings
                select_exprs.append(
                    pl.when(pl.col(col_name).str.len_bytes() > self.max_str)
                    .then(pl.col(col_name).str.slice(0, self.max_str) + '...')
                    .otherwise(pl.col(col_name))
                    .alias(col_name)
                )
            else:
                select_exprs.append(pl.col(col_name))
        
        truncated_data = dataset.data.select(select_exprs)
        return truncated_data.write_csv()
    
    @override
    def format_table(
        self,
        sample_data: Dataset,
    ) -> str:
        """Format schema and sample data for a single table.
        
        Includes markdown headers at the specified level. Formats the schema as a list
        of columns with their DuckDB types, and formats the sample data as CSV.
        
        Args:
            sample_data: Dataset containing the sample data to format

        Returns:
            String representation of the table with its schema and sample data
        """
        markdown_header = '#' * self.markdown_level
        # Schema
        out = [f'{markdown_header} Schema:']
        out.append(self._format_schema(sample_data.schema))
        out.append('')
        
        # Sample data
        out.append(f'{markdown_header} Sample Data:')
        out.append(self._format_sample_data(sample_data))
        out.append('')
        return '\n'.join(out)
    

@attrs.frozen
class MarkdownTablesFormatter(TablesFormatter):
    """Markdown tables formatter.
    
    Formats all available tables (primary and secondary) with their schemas and sample data
    in markdown format. Each table includes a header with its name, followed by its schema
    and sample data sections.
    
    Args:
        markdown_level: The markdown header level to use for table sections (default: 2).
        num_samples: Number of sample rows to retrieve for each table (default: 5).
        max_str: Maximum string length for cell values. Longer values are truncated with '...' (default: 100).
        table_formatter: TableFormatter to use for formatting individual tables.
                        Defaults to MarkdownTableFormatter with markdown_level + 1.
        primary_dataset_sampler: DataSampler to use for sampling the primary dataset.
                                Defaults to RandomSampler for representative sampling.
        tables_sampler: TableSampler to use for sampling the secondary tables.
                       Defaults to HeadTableSampler for consistent sampling.
    
    Example:
        ## Primary Table: users

        ### Schema:
        - id: INTEGER
        - name: VARCHAR
        - tier: ENUM('bronze', 'silver', 'gold')

        ### Sample Data:
        id,name,tier
        1,Alice,gold
        2,Bob,silver

        ## Table: orders

        ### Schema:
        - order_id: INTEGER
        - user_id: INTEGER
        - amount: DOUBLE

        ### Sample Data:
        order_id,user_id,amount
        101,1,49.99
        102,2,125.50
    """
    markdown_level: int = 2
    num_samples: int = 5
    max_str: int = 100
    table_formatter: TableFormatter = attrs.field(default=attrs.Factory(lambda self: MarkdownTableFormatter(markdown_level=self.markdown_level + 1, max_str=self.max_str), takes_self=True))
    primary_dataset_sampler: DataSampler = RandomSampler()
    tables_sampler: TableSampler = HeadTableSampler()

    @override
    def format_all_tables(
        self,
        input: Dataset,
        tables: TablesWithJoinStrategies,
        conn: DuckDBPyConnection,
        random_seed: int | None = None,
    ) -> str:
        """Format all available tables with their schemas and sample data for LLM prompts.
        
        Formats the primary table followed by all secondary tables. Each table is formatted
        with a header at the specified markdown level, followed by its schema and sample data
        using the configured table formatter.

        Args:
            input: Primary input dataset to format.
            tables: Secondary tables with their join strategies to format.
            conn: Database connection for querying sample data from secondary tables.
            random_seed: Optional random seed for reproducible sampling.

        Returns:
            Markdown-formatted string containing all tables with their schemas and sample data.
        """
        sections = []
        markdown_header = '#' * self.markdown_level
        # get sample data for primary table
        sample_data = self.primary_dataset_sampler.sample(input, self.num_samples, random_seed=random_seed)
        # Format primary table
        sections.append(f'{markdown_header} Primary Table: {self.primary_table_name}\n')
        sections.append(self.table_formatter.format_table(sample_data))
        # Format secondary tables
        for table_with_strategies in tables:
            # get sample data for the table
            sample_data = self.tables_sampler.sample(
                table_with_strategies,
                conn,
                self.num_samples,
                random_seed=random_seed,
            )
            # Format table section
            sections.append(f'{markdown_header} Table: {table_with_strategies.table.name.name}\n')
            sections.append(self.table_formatter.format_table(sample_data))
        
        return '\n'.join(sections)
