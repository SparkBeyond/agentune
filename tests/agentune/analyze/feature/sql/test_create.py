
from duckdb import DuckDBPyConnection

from agentune.analyze.feature.sql.base import (
    BoolSqlBackedFeature,
    CategoricalSqlBackedFeature,
    FloatSqlBackedFeature,
    IntSqlBackedFeature,
)
from agentune.analyze.feature.sql.create import feature_from_query
from agentune.core import types
from agentune.core.schema import Field, Schema


def test_feature_from_query(conn: DuckDBPyConnection) -> None:
    conn.execute('CREATE TABLE context_table (key int, value int)')
    conn.execute('INSERT INTO context_table VALUES (1, 2), (3, 4)')

    feature = feature_from_query(conn,
                                 '''select key from my_table''',
                                 Schema((Field('key', types.int32), )),
                                 (),
                                 'my_table')
    assert isinstance(feature, IntSqlBackedFeature)
    assert feature.name == 'key'

    feature = feature_from_query(conn,
                                 '''select key::utinyint as key from my_table''',
                                 Schema((Field('key', types.int32), )),
                                 (),
                                 'my_table')
    assert isinstance(feature, IntSqlBackedFeature)
    assert feature.name == 'key'

    feature = feature_from_query(conn,
                                 '''select key > 1 as foo from my_table''',
                                 Schema((Field('key', types.int32), )),
                                 (),
                                 'my_table')
    assert isinstance(feature, BoolSqlBackedFeature)
    assert feature.name == 'foo'

    feature = feature_from_query(conn,
                                 '''select key::double as key from my_table''',
                                 Schema((Field('key', types.int32), )),
                                 (),
                                 'my_table')
    assert isinstance(feature, FloatSqlBackedFeature)
    assert feature.name == 'key'

    feature = feature_from_query(conn,
                                 '''select key::varchar as key from my_table''',
                                 Schema((Field('key', types.int32), )),
                                 (),
                                 'my_table')
    assert isinstance(feature, CategoricalSqlBackedFeature)
    assert feature.name == 'key'
    assert feature.categories == ('nonesuch',)

    feature = feature_from_query(conn,
                                 '''select key::enum('1', '2') as key from my_table''',
                                 Schema((Field('key', types.int32), )),
                                 (),
                                 'my_table')
    assert isinstance(feature, CategoricalSqlBackedFeature)
    assert feature.name == 'key'
    assert feature.categories == ('1', '2')
