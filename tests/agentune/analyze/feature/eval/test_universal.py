import polars as pl
import pytest
from duckdb.duckdb import DuckDBPyConnection
from tests.agentune.analyze.run.feature_search.toys import ToyAsyncFeature, ToySyncFeature

from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.core.types import float64
from agentune.analyze.feature.eval.universal import (
    UniversalAsyncFeatureEvaluator,
    UniversalSyncFeatureEvaluator,
)


@pytest.fixture
def sample_dataset() -> Dataset:
    schema = Schema((
        Field('a', float64),
        Field('b', float64),
    ))
    data = pl.DataFrame({
        'a': [1.0, 2.0, 3.0],
        'b': [4.0, 5.0, 6.0], 
    })
    return Dataset(schema, data)


@pytest.fixture 
def sync_feature() -> ToySyncFeature:
    return ToySyncFeature('a', 'b', 'sum', 'Sum of a and b', 'a + b')


@pytest.fixture
def async_feature() -> ToyAsyncFeature:
    return ToyAsyncFeature('a', 'b', 'sum', 'Sum of a and b', 'a + b')

def test_universal_sync_supports_sync_features(sync_feature: ToySyncFeature) -> None:
    assert UniversalSyncFeatureEvaluator.supports_feature(sync_feature)

def test_universal_sync_rejects_async_features(async_feature: ToyAsyncFeature) -> None:
    assert not UniversalSyncFeatureEvaluator.supports_feature(async_feature)

def test_universal_async_supports_async_features(async_feature: ToyAsyncFeature) -> None:
    assert UniversalAsyncFeatureEvaluator.supports_feature(async_feature)

def test_universal_async_rejects_sync_features(sync_feature: ToySyncFeature) -> None:
    assert not UniversalAsyncFeatureEvaluator.supports_feature(sync_feature)

def test_universal_sync_evaluator(conn: DuckDBPyConnection, sample_dataset: Dataset,
                                  sync_feature: ToySyncFeature) -> None:
    evaluator = UniversalSyncFeatureEvaluator.for_features([sync_feature])

    result = evaluator.evaluate(sample_dataset, conn)
    assert result.schema.names == ['sum']
    assert result.data['sum'].to_list() == [5.0, 7.0, 9.0]  # [1+4, 2+5, 3+6]

async def test_universal_async_evaluator(conn: DuckDBPyConnection, sample_dataset: Dataset,
                                         async_feature: ToyAsyncFeature) -> None:
    evaluator = UniversalAsyncFeatureEvaluator.for_features([async_feature])

    result = await evaluator.aevaluate(sample_dataset, conn)
    assert result.schema.names == ['sum']
    assert result.data['sum'].to_list() == [5.0, 7.0, 9.0]  # [1+4, 2+5, 3+6]
