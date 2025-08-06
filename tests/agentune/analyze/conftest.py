import contextlib
from collections.abc import AsyncIterator, Iterator

import pytest
from duckdb import DuckDBPyConnection

from agentune.analyze.core.database import DuckdbManager
from agentune.analyze.run.base import RunContext


@pytest.fixture
def ddb_manager() -> Iterator[DuckdbManager]:
    """Provide a DuckdbManager connected to an in-memory database."""
    with contextlib.closing(DuckdbManager.in_memory()) as ddb_manager:
        yield ddb_manager


@pytest.fixture
def conn(ddb_manager: DuckdbManager) -> Iterator[DuckDBPyConnection]:
    """Provide an in-memory DuckDB connection."""
    with ddb_manager.cursor() as conn:
        yield conn

@pytest.fixture
async def run_context(ddb_manager: DuckdbManager) -> AsyncIterator[RunContext]:
    """Create a default RunContext backed by an in-memory DuckDBManager."""
    async with contextlib.aclosing(RunContext.create_default_context(ddb_manager)) as run_context:
        yield run_context
