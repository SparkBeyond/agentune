import duckdb

from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.schema import Schema
from agentune.analyze.join.lookup import LookupJoinStrategy


def test_lookup() -> None:
    with duckdb.connect(':memory:lookup') as conn:
        conn.execute('create table test(key integer, val1 integer, val2 varchar)')

        table = DuckdbTable.from_duckdb('test', conn)
        strategy = LookupJoinStrategy[int]('lookup', table, table.schema['key'], (table.schema['val1'], table.schema['val2']))
        strategy.index.create(conn, 'test')
        
        conn.execute("insert into test values (1, 10, 'a'), (2, 20, 'b'), (3, 30, 'c')")

        assert strategy.get(conn, 1, 'val1') == 10
        assert strategy.get(conn, 1, 'val2') == 'a'
        assert strategy.get(conn, 2, 'val1') == 20
        assert strategy.get(conn, 2, 'val2') == 'b'
        assert strategy.get(conn, 3, 'val1') == 30
        assert strategy.get(conn, 3, 'val2') == 'c'
        assert strategy.get(conn, 4, 'val1') is None
        
        assert strategy.get_many(conn, 1, ['val1', 'val2']) == (10, 'a')
        assert strategy.get_many(conn, 2, ['val1', 'val2']) == (20, 'b')
        assert strategy.get_many(conn, 3, ['val1', 'val2']) == (30, 'c')
        assert strategy.get_many(conn, 4, ['val1', 'val2']) is None

        dataset = strategy.get_batch(conn, [1, 2, 3], ['val1', 'val2'])
        assert dataset.schema == table.schema
        assert dataset.data.to_dicts() == [
            {'key': 1, 'val1': 10, 'val2': 'a'},
            {'key': 2, 'val1': 20, 'val2': 'b'},
            {'key': 3, 'val1': 30, 'val2': 'c'},
        ]
        
        # With a nonexistent key
        dataset2 = strategy.get_batch(conn, [1,2,4], ['val1', 'val2'])
        assert dataset2.schema == dataset.schema
        assert dataset2.data.to_dicts() == [
            {'key': 1, 'val1': 10, 'val2': 'a'},
            {'key': 2, 'val1': 20, 'val2': 'b'},
            {'key': 4, 'val1': None, 'val2': None},
        ]

        # Requesting only some value columns
        dataset2 = strategy.get_batch(conn, [1, 2, 3], ['val1'])
        assert dataset2.schema == Schema((table.schema['key'], table.schema['val1']))
        assert dataset2.data.to_dicts() == [
            {'key': 1, 'val1': 10},
            {'key': 2, 'val1': 20},
            {'key': 3, 'val1': 30},
        ]

        # Reordering the input keys - the output order should match
        dataset3 = strategy.get_batch(conn, [2,1,4,3], ['val1', 'val2'])
        assert dataset3.schema == dataset.schema
        assert dataset3.data.to_dicts() == [
            {'key': 2, 'val1': 20, 'val2': 'b'},
            {'key': 1, 'val1': 10, 'val2': 'a'},
            {'key': 4, 'val1': None, 'val2': None},
            {'key': 3, 'val1': 30, 'val2': 'c'},
        ]

    
if __name__ == '__main__':
    test_lookup()
