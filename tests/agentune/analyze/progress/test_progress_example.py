import asyncio
import datetime
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, override

from attrs import frozen
from duckdb import DuckDBPyConnection
from tests.agentune.analyze.run.feature_search.toys import ToySyncFeature

from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbName, DuckdbTable
from agentune.analyze.core.duckdbio import DuckdbTableSink, DuckdbTableSource
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.feature.base import FloatFeature
from agentune.analyze.feature.eval.base import FeatureEvaluator
from agentune.analyze.feature.eval.universal import (
    UniversalAsyncFeatureEvaluator,
    UniversalSyncFeatureEvaluator,
)
from agentune.analyze.join.base import JoinStrategy
from agentune.analyze.progress.base import root_stage_scope
from agentune.analyze.run.enrich.impl import EnrichRunnerImpl

_logger = logging.getLogger(__name__)

@frozen
class DelayingNonbatchingAsyncFeature(FloatFeature):
    col1: str
    col2: str
    name: str
    description: str
    technical_description: str
    delay_per_row: datetime.timedelta

    # Redeclare attributes with defaults
    default_for_missing: float = 0.0
    default_for_nan: float = 0.0
    default_for_infinity: float = 0.0
    default_for_neg_infinity: float = 0.0

    @property
    @override
    def params(self) -> Schema:
        return Schema((Field(self.col1, types.float64), Field(self.col2, types.float64), ))

    @property
    @override
    def secondary_tables(self) -> list[DuckdbTable]:
        return []

    @property
    @override
    def join_strategies(self) -> list[JoinStrategy]:
        return []

    @override
    async def aevaluate(self, args: tuple[Any, ...],
                        conn: DuckDBPyConnection) -> float:
        await asyncio.sleep(self.delay_per_row.total_seconds())
        return args[0] + args[1]


@asynccontextmanager
async def log_progress_every(interval: datetime.timedelta) -> AsyncGenerator[None, None]:
    """An example (non-production) logger that dumps the current progress state (not the difference from last time)
    to the log every interval.

    To let it capture the progress inside this context scope, it creates a new root stage and then logs its children
    but not the root itself.
    """
    with root_stage_scope('prorgress_manager') as root_scope:
        def log() -> None:
            copy = root_scope.deepcopy()
            for child in copy.children:
                _logger.info(f'Progress snapshot:\n{child}')

        async def logging_loop() -> None:
            while True:
                log()
                await asyncio.sleep(interval.total_seconds())

        logging_task = asyncio.create_task(logging_loop())

        try:
            yield
        finally:
            logging_task.cancel()
            # One last log to capture the last value
            log()

# Ignored, rename to test_xxx to run it
async def ignore_demonstrate_progress_logging(conn: DuckDBPyConnection) -> None:
    async with log_progress_every(datetime.timedelta(seconds=1)):
        conn.execute('CREATE TABLE input(a int, b int)')
        conn.execute('INSERT INTO input SELECT x, y FROM unnest(range(100)) AS t1(x) CROSS JOIN unnest(range(100)) AS t2(y)')

        sync_feature = ToySyncFeature('a', 'b', 'a+b', '', '')
        async_featues = [DelayingNonbatchingAsyncFeature('a', 'b', 'a+b', '', '', datetime.timedelta(milliseconds=1))] * 10
        evaluators: list[type[FeatureEvaluator]] = [UniversalSyncFeatureEvaluator, UniversalAsyncFeatureEvaluator]

        source = DuckdbTableSource(DuckdbTable.from_duckdb('input', conn), batch_size=1000)
        sink = DuckdbTableSink(DuckdbName.qualify('sink', conn))

        runner = EnrichRunnerImpl(max_async_features_eval=5)
        await runner.run_stream([sync_feature, *async_featues], source, sink, evaluators, conn)

