"""Schema formatting for LLM prompts."""

from typing import override

import attrs
from duckdb import DuckDBPyConnection

from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.core.dataset import Dataset
from agentune.core.formatter.base import TablesFormatter
from agentune.core.sampler.base import DataSampler, HeadSampler
from agentune.core.schema import Schema


@attrs.frozen
class SimpleTablesFormatter(TablesFormatter):
    """Simple tables formatter for LLM prompts.
    
    Formats all available tables with their schemas and sample data.
    Each table is formatted with:
    - Table name
    - Schema (list of columns with types)
    - Sample data (CSV format)
    """
    num_samples: int = 5
    sampler: DataSampler = HeadSampler()

    def format_table(self, sample_data: Dataset) -> str:
        """Format schema and sample data for a single table."""
        # Schema
        out = ['### Schema:']
        out.append(self._format_schema(sample_data.schema))
        out.append('')
        
        # Sample data
        out.append(f'### Sample Data ({self.num_samples} rows):')
        out.append(self._format_sample_data(sample_data))
        out.append('')
        return '\n'.join(out)

    @override
    def format_all_tables(
        self,
        input: Dataset,
        tables: TablesWithJoinStrategies,
        conn: DuckDBPyConnection,
        random_seed: int | None = None,
    ) -> str:
        """Format all available tables with their schemas and sample data for LLM prompts.

        Args:
            input: Input dataset (primary table)
            tables: Available tables with their join strategies
            conn: Database connection to query sample data

        Returns:
            String representation of all tables with their schemas and sample data
        """
        sections = []
        
        # get sample data for primary table
        sample_data = self.sampler.sample(input, self.num_samples, random_seed=random_seed)
        # Format primary table
        sections.append(f'## Primary Table: {self.primary_table_name}\n')
        sections.append(self._format_table(input.schema, sample_data))

        # Format secondary tables
        for table_with_strategies in tables:
            # get sample data for the table
            table = table_with_strategies.table
            dataset = table.as_source().to_dataset(conn)
            sample_data = self.sampler.sample(dataset, self.num_samples, random_seed=random_seed)
            # Format table section
            sections.append(f'## Table: {table.name.name}\n')
            sections.append(self._format_table(dataset.schema, sample_data))
        
        return '\n'.join(sections)
