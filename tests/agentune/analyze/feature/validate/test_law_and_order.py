import attrs
import pytest
from duckdb import DuckDBPyConnection

from agentune.analyze.feature.sql.base import SqlBackedFeature, SqlFeatureSpec
from agentune.analyze.feature.sql.create import feature_from_query, int_feature_from_query
from agentune.analyze.feature.validate.base import FeatureValidationError
from agentune.analyze.feature.validate.law_and_order import LawAndOrderValidator
from agentune.core.database import DuckdbTable
from agentune.core.dataset import Dataset, DatasetSource


def _setup(conn: DuckDBPyConnection, query: str) -> tuple[DuckdbTable, DuckdbTable, DuckdbTable, Dataset, SqlBackedFeature]:
    conn.execute('create or replace table table1(key int)')
    conn.execute('create or replace table table2(key int, i int)')
    conn.execute('create or replace table table3(key int, j int)')
    conn.execute('insert into table1 from unnest(range(100))')
    conn.execute('insert into table2 from unnest(range(50)) key, unnest(range(1)) i')
    conn.execute('insert into table3 from unnest(range(20)) key, unnest(range(10)) j')

    table = DuckdbTable.from_duckdb('table1', conn)
    dataset = DatasetSource.from_table(table).to_dataset(conn)
    table2 = DuckdbTable.from_duckdb('table2', conn)
    table3 = DuckdbTable.from_duckdb('table3', conn)

    feature = feature_from_query(conn,  SqlFeatureSpec(sql_query=query),
                                 table.schema,
                                 (table3, table2),
                                 primary_table_name='table1')

    return table, table2, table3, dataset, feature

async def test_valid_feature(conn: DuckDBPyConnection) -> None:
    _, _, _, dataset, feature = _setup(conn,
                                       '''select table1.key + table2.i
                                         from table1
                                         left join table2 on table1.key = table2.key
                                         order by table1.rowid''')

    await LawAndOrderValidator().validate(feature, dataset, conn)

async def test_constant_output(conn: DuckDBPyConnection) -> None:
    for constant in ['1', 'null', "'NaN'::DOUBLE", "'inf'::DOUBLE", "'-inf'::DOUBLE"]:
        _, _, _, dataset, feature = _setup(conn,
                                           f'''select {constant}
                                             from table1
                                             order by table1.rowid''')
        with pytest.raises(FeatureValidationError, match='always returns'):
            await LawAndOrderValidator().validate(feature, dataset, conn)

async def test_wrong_output_size(conn: DuckDBPyConnection) -> None:
    _, _, _, dataset, feature = _setup(conn,
                                       '''select table1.key + table2.i
                                         from table1
                                         inner join table2 on table1.key = table2.key
                                         order by table1.rowid''')
    with pytest.raises(FeatureValidationError, match='wrong number of rows'):
        await LawAndOrderValidator().validate(feature, dataset, conn)


async def test_inconsistent_behavior(conn: DuckDBPyConnection) -> None:
    _, _, _, dataset, feature = _setup(conn,
                                       '''select random()
                                         from table1
                                         order by table1.rowid''')
    with pytest.raises(FeatureValidationError, match='not consistent'):
        await LawAndOrderValidator().validate(feature, dataset, conn)

async def test_access_other_columns(conn: DuckDBPyConnection) -> None:
    tabl1, table2, table3, dataset, feature = _setup(conn,
                                                     '''select table1.key + table2.i
                                                     from table1
                                                     left join table2 on table1.key = table2.key
                                                     order by table1.rowid''')
    feature = attrs.evolve(feature, secondary_tables=(table3,))
    with pytest.raises(FeatureValidationError, match='Binder Error'):
        await LawAndOrderValidator().validate(feature, dataset, conn)

    table2_without_i = attrs.evolve(table2, schema=table2.schema.drop('i'))
    feature = attrs.evolve(feature, secondary_tables=(table2_without_i, table3))
    with pytest.raises(FeatureValidationError, match='Binder Error'):
        await LawAndOrderValidator().validate(feature, dataset, conn)



async def test_order(conn: DuckDBPyConnection) -> None:
    conn.execute('create table primary_table(i int)')
    conn.execute('insert into primary_table values (1), (2), (3)')

    table = DuckdbTable.from_duckdb('primary_table', conn)
    dataset = DatasetSource.from_table(table).to_dataset(conn)

    feature = int_feature_from_query(conn,
                                     SqlFeatureSpec(sql_query='select i from primary_table'),
                                     table.schema,
                                     ())
    assert feature.compute((2, ), conn) == 2, 'Sanity check'

    # Natural ordering of query gives correctly ordered results without explicit order by rowid
    await LawAndOrderValidator().validate(feature, dataset, conn)

    feature2 = attrs.evolve(feature, sql_query = 'select i from primary_table order by i')
    with pytest.raises(FeatureValidationError, match='reordered'):
        await LawAndOrderValidator().validate(feature2, dataset, conn)

    feature3 = attrs.evolve(feature, sql_query = 'select i - lag(i) over () from primary_table order by primary_table.rowid')
    with pytest.raises(FeatureValidationError, match='reordered'):
        await LawAndOrderValidator().validate(feature3, dataset, conn)

async def test_row_by_row(conn: DuckDBPyConnection) -> None:
    _, _, _, dataset, feature = _setup(conn,
                                       '''select (table1.key + count(*) over ())::integer
                                         from table1
                                         order by table1.rowid''')
    with pytest.raises(FeatureValidationError, match='row-by-row'):
        await LawAndOrderValidator().validate(feature, dataset, conn)
