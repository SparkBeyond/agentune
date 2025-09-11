from typing import override

import duckdb
import polars as pl
from attrs import define, frozen
from mypyc.ir.ops import Sequence

from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.dataset import Dataset, DatasetSourceFromDataset, duckdb_to_polars
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.feature.base import IntFeature, SqlQueryFeature, SyncFeature
from agentune.analyze.join.base import (
    JoinStrategy,
)


@define(slots=False)
class SqlBackedFeature[T](SqlQueryFeature, SyncFeature[T]):
    """A feature implemented as a single SQL query.

    This is an implementation example; it is expected to evolve once we start using it.

    The query can address the main table under the name self.main_table_name and the secondary tables under their
    relation names (which are not the same as the join strategy names!)

    Remember to also extend one of the feature type ABCs (IntFeature, etc).
    """

    index_column_name: str = 'row_index_column'
    main_table_name: str = 'main_table'

    @override
    def evaluate_batch(self, input: Dataset, 
                       conn: duckdb.DuckDBPyConnection) -> pl.Series:
        # Separate cursor to register the main table
        with (conn.cursor() as cursor):
            # Need to explicitly order the result to match the original df
            if self.index_column_name in input.data.columns:
                raise ValueError(f'Input data already has a column named {self.index_column_name}')

            # Go through DatasetSourceFromDataset to make the registered relation have the right schema
            input_with_index_data = input.data.with_row_index(self.index_column_name, input.data.width)
            input_with_index_schema = input.schema + Field(self.index_column_name, types.uint32)
            input_with_index = Dataset(input_with_index_schema, input_with_index_data)
            input_relation = DatasetSourceFromDataset(input_with_index).to_duckdb(cursor)
            cursor.register(self.main_table_name, input_relation)
            result = duckdb_to_polars(cursor.sql(self.sql_query))
            if result.width != 1:
                raise ValueError(f'SQL query must return exactly one column but returned {result.width}')
            return result.to_series(0)

@frozen
class IntSqlFeatureForTests(SqlBackedFeature[pl.Int32], IntFeature):
    params: Schema
    secondary_tables: tuple[DuckdbTable, ...]
    sql_query: str

    name: str = 'test_sql_feature'
    description: str = ''
    technical_description: str = ''
    join_strategies: Sequence[JoinStrategy] = ()

    # Redeclare attributes with defaults
    default_for_missing: int = 0
    index_column_name: str = 'row_index_column'
    main_table_name: str = 'main_table'

    def evaluate_batch(self, input: Dataset, 
                       conn: duckdb.DuckDBPyConnection) -> pl.Series:
        series =  super().evaluate_batch(input, conn)
        assert series.dtype == self.dtype.polars_type, f'SQL query must return a column of type {self.dtype.polars_type} but returned {series.dtype}'
        assert series.len() == input.data.height, f'SQL query must return the same number of rows as the input data but returned {series.len()}'
        return series


def test_sql_feature() -> None:
    with duckdb.connect(':memory:TestSqlFeature') as conn:
        conn.execute('CREATE TABLE context_table (key int, value int)')
        conn.execute('INSERT INTO context_table VALUES (1, 2), (3, 4)')

        context_table = DuckdbTable.from_duckdb('context_table', conn)
        feature = IntSqlFeatureForTests(
            params=Schema((Field('key', types.int32), )),
            secondary_tables=(context_table,),
            sql_query='''
            SELECT context_table.value
            FROM main_table 
            LEFT JOIN context_table ON main_table.key = context_table.key
            ORDER BY main_table.row_index_column
            '''
        )
        
        assert feature.evaluate((1, ), conn) == 2
        assert feature.evaluate((3, ), conn) == 4
        assert feature.evaluate((2, ), conn) is None

        # Batch, with some repeated and some missing keys, to test the ordering
        assert feature.evaluate_batch(
            Dataset(feature.params, pl.DataFrame({'key': [3, 2, 1, 3, 1]})), conn).equals(
                pl.Series('test_sql_feature', [4, None, 2, 4, 2]))


if __name__ == '__main__':
    test_sql_feature()
