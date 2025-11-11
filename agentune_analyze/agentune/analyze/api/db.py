from __future__ import annotations

import asyncio
from collections.abc import Mapping

from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.api.base import RunContext
from agentune.analyze.api.data import BoundTable
from agentune.analyze.core.database import DuckdbDatabase, DuckdbName, DuckdbTable
from agentune.analyze.core.dataset import DatasetSource


@frozen
class BoundDb:
    """Methods for accessing and manipulating the duckdb database, bound to a RunContext instance."""
    run_context: RunContext

    def cursor(self) -> DuckDBPyConnection:
        """Return a new connection to the duckdb database.

        A connection can be used as a context manager, returning itself and closing itself when leaving the context.
        """
        return self.run_context._ddb_manager.cursor()

    async def execute(self, query: str, params: object | None = None) -> BoundDb:
        """Immediately execute an SQL statement. For queries, use ctx.data.query().

        For the semantics of `query` and `params`, see https://duckdb.org/docs/stable/clients/python/relational_api#execute
        and https://duckdb.org/docs/stable/clients/python/dbapi#prepared-statements.

        If you want to inspect the result set, you need to work directly with the duckdb connection.
        """
        with self.run_context._ddb_manager.cursor() as conn:
            await asyncio.to_thread(conn.execute, query, params)
        return self

    @property
    def databases(self) -> Mapping[str, DuckdbDatabase]:
        """List the attached duckdb databases, including the main one (i.e. the one opened when the context was created).

        The returned mapping is keyed by the name under which the database is available in the duckdb connection.
        I.e., you can query a table foo.bar.baz where foo is the database (catalog) name returned here and bar is the schema name.
        The names are freely chosen when attaching the databases and are not inherent to them or stored in them.
        """
        return self.run_context._ddb_manager.databases()

    def attach(self, db: DuckdbDatabase, name: str | None = None) -> BoundDb:
        """Attach another database to duckdb under a freely chosen name.

        An in-memory database is created when attached; an on-disk database is created or opened if it already exists.
        Several processes can attach to the same on-disk database only if they all open it in read-only mode.
        """
        self.run_context._ddb_manager.attach(db, name)
        return self

    def detach(self, name: str) -> BoundDb:
        """Detach a previously attached database from duckdb.

        The main database (the one specified when creating the context) cannot be detached.
        In-memory databases are discarded when they are detached.
        """
        self.run_context._ddb_manager.detach(name)
        return self

    def table(self, source: DuckdbName | str | DuckdbTable,
              batch_size: int | None = None) -> BoundTable:
        """Reference an existing table in the duckdb database."""
        batch_size = batch_size or self.run_context.defaults.duckdb_batch_size
        match source:
            case DuckdbTable():
                duckdb_table = source
            case _:
                with self.run_context._ddb_manager.cursor() as conn:
                    duckdb_table = DuckdbTable.from_duckdb(source, conn)

        dataset_source = DatasetSource.from_table(duckdb_table, batch_size)
        return BoundTable(self.run_context, duckdb_table, dataset_source)

