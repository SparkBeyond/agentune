from typing import override

import attrs
import duckdb
import polars as pl

from agentune.analyze.context.base import TablesWithContextDefinitions
from agentune.analyze.core import types
from agentune.analyze.core.dataset import Dataset, DatasetSourceFromDataset, duckdb_to_polars
from agentune.analyze.core.schema import Field
from agentune.analyze.feature.base import SqlQueryFeature, SyncFeature


@attrs.define(slots=False) # for declaring index_column_name
class SqlBackedFeature[T](SqlQueryFeature, SyncFeature[T]):
    """A feature implemented as a single SQL query.

    The query can address the main table under the name 'main_table' and the context tables under their 
    relation names (which are not the same as the context definition names!).
    TODO Context tables can also be named main_table; need to put it in a separate schema.

    Remember that you still have to extend one of the feature type interfaces (IntFeature, etc).
    """

    index_column_name: str = 'row_index_column'

    @override
    def evaluate_batch(self, input: Dataset, contexts: TablesWithContextDefinitions,
                       conn: duckdb.DuckDBPyConnection) -> pl.Series:
        # Separate cursor to register the main table
        with conn.cursor() as cursor:
            # Need to explicitly order the result to match the original df
            if self.index_column_name in input.data.columns:
                raise ValueError(f'Input data already has a column named {self.index_column_name}')

            # Go through DatasetSourceFromDataset to make the registered relation have the right schema
            input_with_index_data = input.data.with_row_index(self.index_column_name, input.data.width)
            input_with_index_schema = input.schema + Field(self.index_column_name, types.uint32)
            input_with_index = Dataset(input_with_index_schema, input_with_index_data)
            input_relation = DatasetSourceFromDataset(input_with_index).to_duckdb(cursor)
            cursor.register('main_table', input_relation)

            # TODO remove asserts from production code
            result = duckdb_to_polars(cursor.sql(self.sql_query))
            assert result.width == 1, f'SQL query must return exactly one column but returned {result.width}'
            series = result.to_series(0)
            assert series.dtype == self.dtype.polars_type, f'SQL query must return a column of type {self.dtype.polars_type} but returned {series.dtype}'
            assert series.len() == input.data.height, f'SQL query must return the same number of rows as the input data but returned {series.len()}'
            return series

