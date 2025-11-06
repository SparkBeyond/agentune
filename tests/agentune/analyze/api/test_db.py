import polars as pl
import pytest
from duckdb import CatalogException

from agentune.analyze.api.base import RunContext
from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbName
from agentune.analyze.core.schema import Field, Schema


async def test_get_table(ctx: RunContext) -> None:
    await ctx.db.execute('create table test(id integer)')
    await ctx.db.execute('insert into test(id) values (1)')

    with pytest.raises(CatalogException, match='does not exist'):
        ctx.db.table('nonesuch')

    table = ctx.db.table('test')
    assert table.name.name == 'test'
    assert table.schema == Schema((Field('id', types.int32),))
    assert (await table.load()).data.equals(pl.DataFrame({'id': [1]}))

    with ctx.db.cursor() as conn:
        assert ctx.db.table(DuckdbName.qualify('test', conn)).name == table.name
    assert ctx.db.table(table.table) == table

