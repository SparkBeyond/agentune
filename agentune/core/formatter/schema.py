"""Schema formatting for LLM prompts."""

from typing import override

import attrs
from duckdb import DuckDBPyConnection

from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.core.dataset import Dataset
from agentune.core.formatter.base import SchemaFormatter
from agentune.core.sampler.base import DataSampler, HeadSampler


@attrs.frozen
class SimpleSchemaFormatter(SchemaFormatter):
    """Simple schema formatter for LLM prompts.
    
    Formats all available tables with their schemas and sample data.
    Each table is formatted with:
    - Table name
    - Schema (list of columns with types)
    - Sample data (CSV format)
    """
    num_samples: int = 5
    sampler: DataSampler = HeadSampler()

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
        
        # Format primary table
        sections.append('## Primary Table: ' + self.primary_table_name)
        sections.append('')
        sections.append('### Schema:')
        sections.append(self._serialize_schema(input.schema))
        sections.append('')
        sections.append(f'### Sample Data ({self.num_samples} rows):')
        sample_data = self.sampler.sample(input, self.num_samples, random_seed=random_seed)
        sections.append(self._format_sample_data(sample_data))
        sections.append('')
        
        # Format secondary tables
        for table_with_strategies in tables:
            table = table_with_strategies.table
            sections.append(f'## Table: {table.name.name}')
            sections.append('')
            
            # Schema
            sections.append('### Schema:')
            sections.append(self._serialize_schema(table.schema))
            sections.append('')
            
            # Sample data
            sections.append(f'### Sample Data ({self.num_samples} rows):')
            dataset = table.as_source().to_dataset(conn)
            sample_data = self.sampler.sample(dataset, self.num_samples, random_seed=random_seed)
            sections.append(self._format_sample_data(sample_data))
            sections.append('')
        
        return '\n'.join(sections)
