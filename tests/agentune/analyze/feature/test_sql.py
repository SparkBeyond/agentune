import duckdb
import polars as pl
from attrs import frozen

from agentune.analyze.context.base import TablesWithContextDefinitions, TableWithContextDefinitions
from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.feature.base import IntFeature
from agentune.analyze.feature.sql import SqlBackedFeature


@frozen
class IntSqlFeatureForTests(SqlBackedFeature[pl.Int32], IntFeature):
    params: Schema
    context_tables: tuple[DuckdbTable, ...]
    sql_query: str
    index_column_name: str = 'row_index_column'

    name = 'test_sql_feature'
    description = ''
    code = ''
    context_objects = ()

def test_sql_feature() -> None:
    with duckdb.connect(':memory:TestSqlFeature') as conn:
        conn.sql('CREATE TABLE context_table (key int, value int)')
        conn.sql('INSERT INTO context_table VALUES (1, 2), (3, 4)')

        context_table = DuckdbTable.from_duckdb('context_table', conn)
        tables_with_contexts = TablesWithContextDefinitions.from_list([
            TableWithContextDefinitions(
                context_table,
                context_definitions=()
            )
        ])
        feature = IntSqlFeatureForTests(
            Schema((Field('key', types.int32), )),
            (context_table,),
            '''
            SELECT context_table.value
            FROM main_table 
            LEFT JOIN context_table ON main_table.key = context_table.key
            ORDER BY main_table.row_index_column
            '''
        )
        
        assert feature.evaluate((1, ), tables_with_contexts, conn) == 2
        assert feature.evaluate((3, ), tables_with_contexts, conn) == 4
        assert feature.evaluate((2, ), tables_with_contexts, conn) is None

        # Batch, with some repeated and some missing keys, to test the ordering
        assert feature.evaluate_batch(
            Dataset(feature.params, pl.DataFrame({'key': [3, 2, 1, 3, 1]})), tables_with_contexts, conn).equals(
                pl.Series('test_sql_feature', [4, None, 2, 4, 2]))


if __name__ == '__main__':
    test_sql_feature()
