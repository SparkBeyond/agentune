import asyncio
import functools
import logging
from collections.abc import Sequence
from typing import Any, override

import polars as pl
import pytest
from attrs import frozen
from duckdb.duckdb import DuckDBPyConnection
from tests.agentune.analyze.run.feature_search.toys import (
    ToyAsyncFeature,
    ToySyncFeature,
)

from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbName, DuckdbTable
from agentune.analyze.core.dataset import Dataset, DatasetSourceFromIterable
from agentune.analyze.core.duckdbio import DuckdbTableSink, DuckdbTableSource
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.core.types import float64
from agentune.analyze.feature.base import FloatFeature
from agentune.analyze.feature.eval.base import FeatureEvaluator
from agentune.analyze.feature.eval.universal import (
    UniversalAsyncFeatureEvaluator,
    UniversalSyncFeatureEvaluator,
)
from agentune.analyze.join.base import JoinStrategy
from agentune.analyze.run.enrich.impl import EnrichRunnerImpl
from agentune.analyze.util.atomic import AtomicInt

_logger = logging.getLogger(__name__)

@frozen
class CountingAsyncFeature(FloatFeature):
    """Adds two float columns together. Sleep for a bit. Count max concurrent feature executions."""

    concurrent_calls: AtomicInt
    max_concurrent_calls: AtomicInt

    col1: str
    col2: str
    name: str
    description: str
    technical_description: str

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
        self.concurrent_calls.inc_and_get()
        await asyncio.sleep(0.001)
        self.max_concurrent_calls.setmax(self.concurrent_calls.get())
        await asyncio.sleep(0.001)
        self.concurrent_calls.inc_and_get(-1)

        return args[0] + args[1]


@pytest.fixture
def sample_dataset() -> Dataset:
    """Create a sample dataset for testing."""
    schema = Schema((
        Field('a', float64),
        Field('b', float64),
        Field('c', float64),
    ))
    data = pl.DataFrame({
        'a': [1.0, 2.0, 3.0],
        'b': [4.0, 5.0, 6.0], 
        'c': [7.0, 8.0, 9.0],
    })
    return Dataset(schema, data)


@pytest.fixture
def sync_features() -> list[ToySyncFeature]:
    """Create sync features that add two columns."""
    return [
        ToySyncFeature('a', 'b', 'a_plus_b', 'Adds a and b', 'a + b'),
        ToySyncFeature('a', 'c', 'a_plus_c', 'Adds a and c', 'a + c'),
    ]


@pytest.fixture
def async_features() -> list[ToyAsyncFeature]:
    """Create async features that add two columns."""
    return [
        ToyAsyncFeature('a', 'b', 'a_plus_b', 'Async adds a and b', 'a + b'),
        ToyAsyncFeature('b', 'c', 'b_plus_c', 'Async adds b and c', 'b + c'),
    ]


async def test_run(conn: DuckDBPyConnection, sample_dataset: Dataset,
                   sync_features: Sequence[ToySyncFeature], async_features: Sequence[ToyAsyncFeature]) -> None:
    runner = EnrichRunnerImpl()
    evaluators: list[type[FeatureEvaluator]] = [UniversalSyncFeatureEvaluator, UniversalAsyncFeatureEvaluator]

    expected_a_plus_b = [5.0, 7.0, 9.0]  # [1+4, 2+5, 3+6]
    expected_a_plus_c = [8.0, 10.0, 12.0]  # [1+7, 2+8, 3+9]
    expected_b_plus_c = [11.0, 13.0, 15.0]  # [4+7, 5+8, 6+9]

    sync_result = await runner.run(sync_features, sample_dataset, evaluators, conn)
    assert sync_result.data.equals(pl.DataFrame({
        'a_plus_b': expected_a_plus_b,
        'a_plus_c': expected_a_plus_c
    }))

    async_result = await runner.run(async_features, sample_dataset, evaluators, conn)
    assert async_result.data.equals(pl.DataFrame({
        'a_plus_b': expected_a_plus_b,
        'b_plus_c': expected_b_plus_c
    }))

    mixed_result = await runner.run([*sync_features, *async_features], sample_dataset, evaluators, conn)
    assert mixed_result.data.equals(pl.DataFrame({
        'a_plus_b': expected_a_plus_b,
        'a_plus_c': expected_a_plus_c,
        'a_plus_b_': expected_a_plus_b,
        'b_plus_c': expected_b_plus_c
    }))

    mixed_result2 = await runner.run([sync_features[0], async_features[0], sync_features[1], async_features[1]],
                                      sample_dataset, evaluators, conn)
    assert mixed_result2.data.equals(pl.DataFrame({
        'a_plus_b': expected_a_plus_b,
        'a_plus_b_': expected_a_plus_b,
        'a_plus_c': expected_a_plus_c,
        'b_plus_c': expected_b_plus_c
    })), 'Output column order must match input feature order, regardless of division into evaluators'

    with pytest.raises(ValueError, match='No evaluator found for features'):
        await runner.run([*async_features, *sync_features], sample_dataset, [UniversalSyncFeatureEvaluator], conn)


async def test_run_with_duplicate_names(conn: DuckDBPyConnection,
                                        sample_dataset: Dataset) -> None:
    """Test run() method with duplicate feature names and deduplication enabled."""
    runner = EnrichRunnerImpl()
    evaluators = [UniversalSyncFeatureEvaluator]

    # Create features with duplicate names
    features = [
        ToySyncFeature('a', 'b', 'sum', 'First sum', 'a + b'),
        ToySyncFeature('a', 'c', 'sum', 'Second sum', 'a + c'),
    ]

    result = await runner.run(features, sample_dataset, evaluators, conn)

    # Names should be deduplicated
    assert result.schema.names == ['sum', 'sum_']
    assert result.data.height == 3

    # With dedup disabled, this fails
    with pytest.raises(ValueError, match='Duplicate feature names found'):
        await runner.run(features, sample_dataset, evaluators, conn, deduplicate_names=False)


async def test_run_with_no_evaluator_for_feature(conn: DuckDBPyConnection,
                                                 sample_dataset: Dataset,
                                                 sync_features: list[ToySyncFeature]) -> None:
    """Test run() method when no evaluator can handle a feature."""
    runner = EnrichRunnerImpl()
    # Only provide async evaluator for sync features
    evaluators = [UniversalAsyncFeatureEvaluator]

    with pytest.raises(ValueError, match='No evaluator found for features'):
        await runner.run(sync_features, sample_dataset, evaluators, conn)

async def test_run_stream(conn: DuckDBPyConnection, sample_dataset: Dataset,
                          sync_features: list[ToySyncFeature], async_features: list[ToyAsyncFeature]) -> None:
    runner = EnrichRunnerImpl()
    features = [*sync_features, *async_features]
    evaluators: list[type[FeatureEvaluator]] = [UniversalSyncFeatureEvaluator, UniversalAsyncFeatureEvaluator]

    # Create a dataset source with multiple datasets
    sample_datasets = [sample_dataset] * 5
    dataset_source = DatasetSourceFromIterable(sample_dataset.schema, sample_datasets)

    enriched_dataset = await runner.run(features, sample_dataset, evaluators, conn)
    dataset_sink = DuckdbTableSink(DuckdbName.qualify('sink', conn))
    await runner.run_stream(features, dataset_source, dataset_sink, evaluators, conn)

    result_table = DuckdbTable.from_duckdb('sink', conn)
    result_dataset = DuckdbTableSource(result_table).to_dataset(conn)
    assert result_table.schema == enriched_dataset.schema
    assert len(result_dataset) == len(sample_datasets) * len(enriched_dataset)
    assert result_dataset == functools.reduce(Dataset.vstack, [enriched_dataset] * 5)


async def test_empty_features_list(conn: DuckDBPyConnection,
                                   sample_dataset: Dataset) -> None:
    """Test run() method with empty features list."""
    runner = EnrichRunnerImpl()
    evaluators = [UniversalSyncFeatureEvaluator]

    result = await runner.run([], sample_dataset, evaluators, conn)

    assert result.schema.names == []
    assert result.data.height == 0
    assert result.data.width == 0

async def test_larger_data_streaming(conn: DuckDBPyConnection) -> None:
    conn.execute('CREATE TABLE input(a int, b int)')
    conn.execute('INSERT INTO input SELECT x, y FROM unnest(range(100)) AS t1(x) CROSS JOIN unnest(range(100)) AS t2(y)')

    feature1 = ToySyncFeature('a', 'b', 'a+b', '', '')
    feature2 = ToyAsyncFeature('a', 'b', 'a+b', '', '')
    evaluators: list[type[FeatureEvaluator]] = [UniversalSyncFeatureEvaluator, UniversalAsyncFeatureEvaluator]

    source = DuckdbTableSource(DuckdbTable.from_duckdb('input', conn), batch_size=1000)
    sink = DuckdbTableSink(DuckdbName.qualify('sink', conn))

    runner = EnrichRunnerImpl()
    await runner.run_stream([feature1, feature2], source, sink, evaluators, conn)

    sink_table = DuckdbTable.from_duckdb('sink', conn)
    sink_dataset = DuckdbTableSource(sink_table).to_dataset(conn)
    assert sink_dataset.schema.names == ['a+b', 'a+b_']
    assert sink_dataset.schema.dtypes == [types.float64, types.float64]

    expected_sums = pl.Series(float(x+y) for x in range(100) for y in range(100)).sort()
    assert sink_dataset.data['a+b'].sort().equals(expected_sums)
    assert sink_dataset.data['a+b_'].sort().equals(expected_sums)

async def test_keep_input_cols(conn: DuckDBPyConnection) -> None:
    conn.execute('CREATE TABLE input(a int, b int)')
    conn.execute('INSERT INTO input SELECT x, y FROM unnest(range(100)) AS t1(x) CROSS JOIN unnest(range(100)) AS t2(y)')

    feature1 = ToySyncFeature('a', 'b', 'a+b', '', '')
    feature2 = ToyAsyncFeature('a', 'b', 'a+b', '', '')
    evaluators: list[type[FeatureEvaluator]] = [UniversalSyncFeatureEvaluator, UniversalAsyncFeatureEvaluator]

    source = DuckdbTableSource(DuckdbTable.from_duckdb('input', conn), batch_size=1000)
    sink = DuckdbTableSink.into_unqualified_duckdb_table('sink', conn)

    runner = EnrichRunnerImpl()

    source_dataset = source.to_dataset(conn)
    enriched_dataset = await runner.run([feature1, feature2], source_dataset,
                                        evaluators, conn,
                                        keep_input_columns=['a'])
    assert enriched_dataset.schema.names == ['a', 'a+b', 'a+b_']
    assert enriched_dataset.schema.dtypes == [types.int32, types.float64, types.float64]
    assert enriched_dataset.data['a'].equals(source_dataset.data['a'])

    await runner.run_stream([feature1, feature2], source, sink, evaluators, conn,
                            keep_input_columns=['a'])

    sink_table = DuckdbTable.from_duckdb('sink', conn)
    sink_dataset = DuckdbTableSource(sink_table).to_dataset(conn)
    assert sink_dataset.schema.names == ['a', 'a+b', 'a+b_']
    assert sink_dataset.schema.dtypes == [types.int32, types.float64, types.float64]

    expected_sums = pl.Series(float(x+y) for x in range(100) for y in range(100)).sort()
    assert sink_dataset.data['a'].equals(source.select('a').to_dataset(conn).data['a'])
    assert sink_dataset.data['a+b'].sort().equals(expected_sums)
    assert sink_dataset.data['a+b_'].sort().equals(expected_sums)

async def test_limits(conn: DuckDBPyConnection, sample_dataset: Dataset) -> None:
    for _ in range(6):
        sample_dataset = sample_dataset.vstack(sample_dataset) # 3*2^6 rows

    concurrent_calls = AtomicInt()
    max_concurrent_calls = AtomicInt()
    features = [
        CountingAsyncFeature(concurrent_calls, max_concurrent_calls, 'a', 'b', 'a_plus_b', 'Async adds a and b', 'a + b'),
        CountingAsyncFeature(concurrent_calls, max_concurrent_calls, 'b', 'c', 'b_plus_c', 'Async adds b and c', 'b + c'),
    ]

    runner = EnrichRunnerImpl(max_async_features_eval=10)
    evaluators: list[type[FeatureEvaluator]] = [UniversalSyncFeatureEvaluator, UniversalAsyncFeatureEvaluator]

    await runner.run(features, sample_dataset, evaluators, conn)

    assert max_concurrent_calls.get() == 10
