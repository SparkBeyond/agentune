import contextlib
from typing import cast

import attrs
import pytest
from duckdb import BinderException, CatalogException

from agentune.analyze.core.database import ArtIndex, DatabaseTable
from agentune.analyze.flow.duckdb import DuckdbManager


def test_tables_indexes() -> None:
    with contextlib.closing(DuckdbManager('conn')) as ddb_manager, ddb_manager.cursor() as conn:
        conn.execute('CREATE TABLE tab (a INT, "quoted name" INT)')
        conn.execute('CREATE INDEX idx ON tab (a, "quoted name")')
        table = DatabaseTable.from_duckdb('tab', conn)
        assert table.indexes == (ArtIndex(name='idx', cols=('a', 'quoted name')),)

        new_index = attrs.evolve(cast(ArtIndex, table.indexes[0]), name='idx2')
        table2 = attrs.evolve(table, name='tab2', indexes=(new_index,))
        table2.create(conn)
        assert DatabaseTable.from_duckdb('tab2', conn) == table2

        table3 = attrs.evolve(table, schema=table.schema.drop('a'))
        table3.create(conn, if_not_exists=True)
        assert DatabaseTable.from_duckdb('tab', conn) == table # Did not replace
        with pytest.raises(CatalogException, match='already exists'):
            table3.create(conn)
        with pytest.raises(BinderException, match='does not have a column named "a"'):
            table3.create(conn, or_replace=True)
        
        table4 = attrs.evolve(table3, indexes=())
        table4.create(conn, or_replace=True)
        assert DatabaseTable.from_duckdb('tab', conn) == table4

