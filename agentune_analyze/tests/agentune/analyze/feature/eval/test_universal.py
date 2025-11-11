import polars as pl
import pytest
from duckdb import DuckDBPyConnection
from tests.agentune.analyze.run.analysis.toys import ToyAsyncFeature, ToySyncFeature

from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.core.types import float64
from agentune.analyze.feature.compute.universal import (
    UniversalAsyncFeatureComputer,
    UniversalSyncFeatureComputer,
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
    assert UniversalSyncFeatureComputer.supports_feature(sync_feature)

def test_universal_sync_rejects_async_features(async_feature: ToyAsyncFeature) -> None:
    assert not UniversalSyncFeatureComputer.supports_feature(async_feature)

def test_universal_async_supports_async_features(async_feature: ToyAsyncFeature) -> None:
    assert UniversalAsyncFeatureComputer.supports_feature(async_feature)

def test_universal_async_rejects_sync_features(sync_feature: ToySyncFeature) -> None:
    assert not UniversalAsyncFeatureComputer.supports_feature(sync_feature)

def test_universal_sync_computer(conn: DuckDBPyConnection, sample_dataset: Dataset,
                                 sync_feature: ToySyncFeature) -> None:
    computer = UniversalSyncFeatureComputer.for_features([sync_feature])

    result = computer.compute(sample_dataset, conn)
    assert result.schema.names == ['sum']
    assert result.data['sum'].to_list() == [5.0, 7.0, 9.0]  # [1+4, 2+5, 3+6]

async def test_universal_async_computer(conn: DuckDBPyConnection, sample_dataset: Dataset,
                                        async_feature: ToyAsyncFeature) -> None:
    computer = UniversalAsyncFeatureComputer.for_features([async_feature])

    result = await computer.acompute(sample_dataset, conn)
    assert result.schema.names == ['sum']
    assert result.data['sum'].to_list() == [5.0, 7.0, 9.0]  # [1+4, 2+5, 3+6]
