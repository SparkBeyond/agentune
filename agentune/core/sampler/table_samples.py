
import random
from typing import override

import attrs
from duckdb import DuckDBPyConnection

from agentune.analyze.join.base import TableWithJoinStrategies
from agentune.core.dataset import Dataset, duckdb_to_dataset
from agentune.core.sampler.base import TableSampler


@attrs.define
class HeadTableSampler(TableSampler):
    """Table sampler that returns the first N rows from a table.
    
    This sampler executes a simple SELECT * query with a LIMIT clause
    to retrieve the head of the table. The order of rows is determined
    by the table's natural order (or index if present).
    """

    @override
    def sample(self, table: TableWithJoinStrategies, conn: DuckDBPyConnection, sample_size: int, random_seed: int | None = None) -> Dataset:
        """Sample the first N rows from the table.
        
        Args:
            table: The table with join strategies to sample from
            conn: The DuckDB connection
            sample_size: Number of rows to sample (limit)
            random_seed: Not used in this sampler (kept for interface compatibility)
            
        Returns:
            Dataset containing the first sample_size rows
        """
        table_name = str(table.table.name)
        sql_query = f'SELECT * FROM {table_name} LIMIT {sample_size}'
        
        relation = conn.sql(sql_query)
        return duckdb_to_dataset(relation)


@attrs.define
class RandomStartTableSampler(TableSampler):
    """Table sampler that returns consecutive rows starting from a random position.
    
    This sampler first determines the table size, then selects a random starting
    point (using the provided seed for reproducibility), and returns consecutive
    rows from that point. Uses row_number() to handle the offset.
    """

    @override
    def sample(self, table: TableWithJoinStrategies, conn: DuckDBPyConnection, sample_size: int, random_seed: int | None = None) -> Dataset:
        """Sample consecutive rows from a random starting point in the table.
        
        Args:
            table: The table with join strategies to sample from
            conn: The DuckDB connection
            sample_size: Number of consecutive rows to sample
            random_seed: Random seed for selecting the starting point (for reproducibility)
            
        Returns:
            Dataset containing sample_size consecutive rows starting from a random position
        """
        table_name = str(table.table.name)
        
        # Get table size
        count_query = f'SELECT COUNT(*) as count FROM {table_name}'
        result = conn.sql(count_query).fetchone()
        table_size = result[0] if result else 0
        
        # Adjust sample size if it exceeds table size
        sample_size = min(sample_size, table_size)
        
        # Select random starting point
        rng = random.Random(random_seed)
        start_rowid = rng.randint(0, max(0, table_size - sample_size))
        
        # Select consecutive rows starting from the random rowid
        # Using DuckDB's built-in rowid pseudocolumn for deterministic and efficient filtering
        sql_query = f'SELECT * FROM {table_name} WHERE rowid >= {start_rowid} LIMIT {sample_size}'
        
        relation = conn.sql(sql_query)
        return duckdb_to_dataset(relation)
