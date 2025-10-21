import logging

import pytest
from duckdb.duckdb import BinderException, ConversionException, DuckDBPyConnection

from agentune.analyze.core.schema import Schema
from agentune.analyze.run.ingest import sampling

_logger = logging.getLogger(__name__)

def test_split_duckdb_table(conn: DuckDBPyConnection) -> None:
    total_count = 1000000

    conn.execute('CREATE TABLE tab(i int)')
    conn.execute(f'INSERT INTO tab SELECT t.i FROM unnest(range({total_count})) AS t(i)')
    orig_schema = Schema.from_duckdb(conn.table('tab'))

    split_table = sampling.split_duckdb_table(conn, 'tab')

    assert split_table.train().cheap_size(conn) == split_table.train().to_dataset(conn).height
    assert split_table.test().cheap_size(conn) == split_table.test().to_dataset(conn).height
    assert split_table.feature_search().cheap_size(conn) == split_table.feature_search().to_dataset(conn).height
    assert split_table.feature_eval().cheap_size(conn) == split_table.feature_eval().to_dataset(conn).height

    num_train = len(conn.table('tab').filter('_is_train'))
    expected_num_train = total_count * 0.8
    assert abs(1 - abs(num_train / expected_num_train)) < 0.001, f'Approximately 0.8 of rows marked as train, {num_train=}'

    num_feature_search = len(conn.table('tab').filter('_is_feature_search'))
    expected_num_feature_search = 10000
    assert num_feature_search == expected_num_feature_search, 'Exactly 10000 rows marked as feature_search'

    num_feature_eval = len(conn.table('tab').filter('_is_feature_eval'))
    expected_num_feature_eval = 100000
    assert num_feature_eval == expected_num_feature_eval, 'Exactly 100000 rows marked as feature_eval'

    train_df = split_table.train().to_dataset(conn)
    test_df = split_table.test().to_dataset(conn)
    feature_search_df = split_table.feature_search().to_dataset(conn)
    feature_eval_df = split_table.feature_eval().to_dataset(conn)

    assert train_df.schema == orig_schema
    assert train_df.data.columns == ['i'] # Regression test for silly bug where we returned too many columns
    assert test_df.schema == orig_schema
    assert test_df.data.columns == ['i']
    assert feature_search_df.schema == orig_schema
    assert feature_search_df.data.columns == ['i']
    assert feature_eval_df.schema == orig_schema
    assert feature_eval_df.data.columns == ['i']

    split_table.drop_split_columns(conn)
    with pytest.raises(BinderException):
        conn.table('tab').filter('_is_train')

    # Test stability - do we select the same rows every time?

    split_table = sampling.split_duckdb_table(conn, 'tab')

    train_df2 = split_table.train().to_dataset(conn)
    test_df2 = split_table.test().to_dataset(conn)
    feature_search_df2 = split_table.feature_search().to_dataset(conn)
    feature_eval_df2 = split_table.feature_eval().to_dataset(conn)

    assert train_df.data.equals(train_df2.data)
    assert test_df.data.equals(test_df2.data)
    assert feature_search_df.data.equals(feature_search_df2.data)
    assert feature_eval_df.data.equals(feature_eval_df2.data)

def test_split_duckdb_table_only_train(conn: DuckDBPyConnection) -> None:
    total_count = 1000000
    conn.execute('CREATE TABLE tab(i int)')
    conn.execute(f'INSERT INTO tab SELECT t.i FROM unnest(range({total_count})) AS t(i)')

    split_table = sampling.split_duckdb_table(conn, 'tab', train_fraction=1.0)

    assert split_table.train().to_dataset(conn).height == total_count
    assert split_table.test().to_dataset(conn).height == 0


def test_split_duckdb_table_error_recovery(conn: DuckDBPyConnection) -> None:
    total_count = 1000000
    conn.execute('CREATE TABLE tab(i int)')
    conn.execute(f'INSERT INTO tab SELECT t.i FROM unnest(range({total_count})) AS t(i)')
    schema = Schema.from_duckdb(conn.table('tab'))

    # Pass a string instead of int to trigger an error halfway through, after the first transaction completes
    with pytest.raises(ConversionException):
        sampling.split_duckdb_table(conn, 'tab', feature_search_size='foo') # type: ignore[arg-type]

    assert Schema.from_duckdb(conn.table('tab')) == schema, 'Table was restored to original state'
