
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
    
    This sampler executes a SELECT * query with a WHERE clause filtering by rowid
    to retrieve the head of the table. Rows are ordered by DuckDB's rowid pseudocolumn.
    """

    @override
    def sample(self, table: TableWithJoinStrategies, conn: DuckDBPyConnection, sample_size: int, random_seed: int | None = 42) -> Dataset:
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
        end_rowid = sample_size - 1
        sql_query = f'SELECT * FROM {table_name} WHERE rowid BETWEEN 0 AND {end_rowid} ORDER BY rowid'
        
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
    def sample(self, table: TableWithJoinStrategies, conn: DuckDBPyConnection, sample_size: int, random_seed: int | None = 42) -> Dataset:
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
        table_size = len(conn.table(table_name))
        
        # Adjust sample size if it exceeds table size
        sample_size = min(sample_size, table_size)
        
        # Select random starting point
        rng = random.Random(random_seed)
        start_rowid = rng.randint(0, max(0, table_size - sample_size))
        end_rowid = start_rowid + sample_size - 1
        
        # Select consecutive rows starting from the random rowid
        # Using DuckDB's built-in rowid pseudocolumn for deterministic and efficient filtering
        sql_query = f'SELECT * FROM {table_name} WHERE rowid BETWEEN {start_rowid} AND {end_rowid} ORDER BY rowid'
        
        relation = conn.sql(sql_query)
        return duckdb_to_dataset(relation)
