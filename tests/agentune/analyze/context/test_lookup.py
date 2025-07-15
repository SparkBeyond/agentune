import duckdb

from agentune.analyze.context.lookup import LookupContextDefinition, LookupContextImpl
from agentune.analyze.core.database import DatabaseTable
from agentune.analyze.core.schema import Schema


def test_lookup() -> None:
    with duckdb.connect(':memory:lookup') as conn:
        conn.execute('create table test(key integer, val1 integer, val2 varchar)')

        table = DatabaseTable.from_duckdb('test', conn)
        context_definition = LookupContextDefinition('lookup', table, table.schema['key'], (table.schema['val1'], table.schema['val2']))
        context_definition.index.create(conn, 'test')
        
        conn.execute("insert into test values (1, 10, 'a'), (2, 20, 'b'), (3, 30, 'c')")

        context = LookupContextImpl[int](conn, context_definition)

        assert context.get(1, 'val1') == 10
        assert context.get(1, 'val2') == 'a'
        assert context.get(2, 'val1') == 20
        assert context.get(2, 'val2') == 'b'
        assert context.get(3, 'val1') == 30
        assert context.get(3, 'val2') == 'c'
        assert context.get(4, 'val1') is None
        
        assert context.get_many(1, ['val1', 'val2']) == (10, 'a')
        assert context.get_many(2, ['val1', 'val2']) == (20, 'b')
        assert context.get_many(3, ['val1', 'val2']) == (30, 'c')
        assert context.get_many(4, ['val1', 'val2']) is None

        dataset = context.get_batch([1, 2, 3], ['val1', 'val2'])
        assert dataset.schema == table.schema
        assert dataset.data.to_dicts() == [
            {'key': 1, 'val1': 10, 'val2': 'a'},
            {'key': 2, 'val1': 20, 'val2': 'b'},
            {'key': 3, 'val1': 30, 'val2': 'c'},
        ]
        
        # With a nonexistent key
        dataset2 = context.get_batch([1,2,4], ['val1', 'val2'])
        assert dataset2.schema == dataset.schema
        assert dataset2.data.to_dicts() == [
            {'key': 1, 'val1': 10, 'val2': 'a'},
            {'key': 2, 'val1': 20, 'val2': 'b'},
            {'key': 4, 'val1': None, 'val2': None},
        ]

        # Requesting only some value columns
        dataset2 = context.get_batch([1, 2, 3], ['val1'])
        assert dataset2.schema == Schema((table.schema['key'], table.schema['val1']))
        assert dataset2.data.to_dicts() == [
            {'key': 1, 'val1': 10},
            {'key': 2, 'val1': 20},
            {'key': 3, 'val1': 30},
        ]

        # Reordering the input keys - the output order should match
        dataset3 = context.get_batch([2,1,4,3], ['val1', 'val2'])
        assert dataset3.schema == dataset.schema
        assert dataset3.data.to_dicts() == [
            {'key': 2, 'val1': 20, 'val2': 'b'},
            {'key': 1, 'val1': 10, 'val2': 'a'},
            {'key': 4, 'val1': None, 'val2': None},
            {'key': 3, 'val1': 30, 'val2': 'c'},
        ]

    
if __name__ == '__main__':
    test_lookup()
