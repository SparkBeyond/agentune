import logging
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, override

import polars as pl
import pytest
from attrs import frozen
from duckdb import DuckDBPyConnection
from tests.agentune.analyze.run.feature_search.toys import (
    ToyAllFeatureSelector,
    ToyAsyncEnrichedFeatureSelector,
    ToyAsyncFeatureGenerator,
    ToyAsyncFeatureSelector,
    ToyPrebuiltFeaturesGenerator,
    ToySyncEnrichedFeatureSelector,
    ToySyncFeature,
    ToySyncFeatureGenerator,
    ToySyncFeatureSelector,
)

from agentune.analyze.context.base import ContextDefinition, TablesWithContextDefinitions
from agentune.analyze.core.database import DuckdbManager, DuckdbName, DuckdbTable
from agentune.analyze.core.dataset import Dataset, DatasetSource
from agentune.analyze.core.duckdbio import (
    DuckdbTableSink,
)
from agentune.analyze.core.schema import Schema
from agentune.analyze.feature.base import (
    BoolFeature,
    CategoricalFeature,
    Classification,
    Feature,
    FloatFeature,
    IntFeature,
    Regression,
)
from agentune.analyze.feature.dedup_names import deduplicate_feature_names
from agentune.analyze.feature.gen.base import GeneratedFeature, SyncFeatureGenerator
from agentune.analyze.feature.select.base import EnrichedFeatureSelector, FeatureSelector
from agentune.analyze.run.base import RunContext
from agentune.analyze.run.feature_search.base import (
    ClassificationFeatureSearchParams,
    FeatureSearchInputData,
    FeatureSearchRunner,
    RegressionFeatureSearchParams,
)
from agentune.analyze.run.feature_search.impl import FeatureSearchRunnerImpl
from agentune.analyze.run.ingest import sampling

_logger = logging.getLogger(__name__)


@frozen
class SimpleFloatFeature(FloatFeature):
    name: str
    description: str = ''
    code: str = ''
    default_for_missing: float = 0.0
    default_for_nan: float = 0.0
    default_for_infinity: float = 0.0
    default_for_neg_infinity: float = 0.0

    params: Schema = Schema(())
    context_tables: tuple[DuckdbTable, ...] = ()
    context_definitions: tuple[ContextDefinition, ...] = ()

    @override
    async def aevaluate(self, args: tuple[Any, ...], contexts: TablesWithContextDefinitions,
                        conn: DuckDBPyConnection) -> float | None:
        return 0.0

@frozen
class SimpleIntFeature(IntFeature):
    name: str
    description: str = ''
    code: str = ''
    default_for_missing: int = 0

    params: Schema = Schema(())
    context_tables: tuple[DuckdbTable, ...] = ()
    context_definitions: tuple[ContextDefinition, ...] = ()

    @override
    async def aevaluate(self, args: tuple[Any, ...], contexts: TablesWithContextDefinitions,
                        conn: DuckDBPyConnection) -> int | None:
        return 0


@frozen
class SimpleBoolFeature(BoolFeature):
    name: str
    description: str = ''
    code: str = ''
    default_for_missing: bool = True

    params: Schema = Schema(())
    context_tables: tuple[DuckdbTable, ...] = ()
    context_definitions: tuple[ContextDefinition, ...] = ()

    @override
    async def aevaluate(self, args: tuple[Any, ...], contexts: TablesWithContextDefinitions,
                        conn: DuckDBPyConnection) -> bool | None:
        return True

@frozen
class SimpleCategoricalFeature(CategoricalFeature):
    name: str
    description: str = ''
    code: str = ''
    default_for_missing: str = CategoricalFeature.other_category

    categories: tuple[str, ...] = ('a', 'b', 'c')

    params: Schema = Schema(())
    context_tables: tuple[DuckdbTable, ...] = ()
    context_definitions: tuple[ContextDefinition, ...] = ()

    @override
    async def aevaluate(self, args: tuple[Any, ...], contexts: TablesWithContextDefinitions,
                        conn: DuckDBPyConnection) -> str | None:
        return None

@frozen
class SimplePrebuiltFeaturesGenerator(SyncFeatureGenerator[Feature]):
    features: tuple[GeneratedFeature, ...]

    @override
    def generate(self, feature_search: Dataset, target_column: str, contexts: TablesWithContextDefinitions,
                 conn: DuckDBPyConnection) -> Iterator[GeneratedFeature]:
        yield from self.features



@pytest.fixture
def input_data_csv_path(tmp_path: Path) -> Path:
    csv_path: Path = tmp_path / 'test.csv'
    df = pl.DataFrame({
        'x': [float(x % 10) for x in range(1, 1000)],
        'y': [float(y % 7) for y in range(1, 1000)],
        'z': [float(z % 5) for z in range(1, 1000)],
        'target': [int(t % 3) for t in range(1, 1000)],
    })
    df.write_csv(csv_path)
    return csv_path

@pytest.fixture
def input_data(input_data_csv_path: Path, ddb_manager: DuckdbManager) -> FeatureSearchInputData:
    with ddb_manager.cursor() as conn:
        csv_input: DatasetSource = DatasetSource.from_csv(input_data_csv_path, conn)
        table_name = DuckdbName.qualify(input_data_csv_path.name, conn)
        DuckdbTableSink(table_name).write(csv_input, conn)
        table = DuckdbTable.from_duckdb(table_name, conn)
        split = sampling.split_duckdb_table(conn, table.name)
        input_data = FeatureSearchInputData.from_split_table(split, 'target', TablesWithContextDefinitions.from_list([]), conn)
        return input_data

async def test_feature_search_regression(input_data: FeatureSearchInputData, run_context: RunContext) -> None:
    # Limit the batch size to make sure multiple batches are tested
    feature_search_runner: FeatureSearchRunner[Regression] = FeatureSearchRunnerImpl(max_features_enrich_batch_size=5)

    selectors: list[FeatureSelector | EnrichedFeatureSelector] = [ToySyncFeatureSelector(), ToyAsyncFeatureSelector(),
                                                                  ToySyncEnrichedFeatureSelector(), ToyAsyncEnrichedFeatureSelector()]
    for selector in selectors:
        _logger.info(f'Running with selector {selector}')
        params = RegressionFeatureSearchParams(
            generators=(ToySyncFeatureGenerator(), ToyAsyncFeatureGenerator()),
            selector=selector
        )
        results = await feature_search_runner.run(run_context, input_data, params)
        _logger.info(results)

async def test_feature_search_classification(input_data: FeatureSearchInputData, run_context: RunContext) -> None:
    feature_search_runner: FeatureSearchRunner[Classification] = FeatureSearchRunnerImpl(max_features_enrich_batch_size=5)

    selectors: list[FeatureSelector | EnrichedFeatureSelector] = [ToySyncFeatureSelector(), ToyAsyncFeatureSelector(),
                                                                  ToySyncEnrichedFeatureSelector(), ToyAsyncEnrichedFeatureSelector()]
    for selector in selectors:
        _logger.info(f'Running with selector {selector}')
        params = ClassificationFeatureSearchParams(
            generators=(ToySyncFeatureGenerator(), ToyAsyncFeatureGenerator()),
            selector=selector
        )
        results = await feature_search_runner.run(run_context, input_data, params)
        _logger.info(results)

async def _test_feature_name_collision(input_data: FeatureSearchInputData, run_context: RunContext,
                                       features: tuple[Feature, ...]) -> None:
    feature_search_runner: FeatureSearchRunner[Regression] = FeatureSearchRunnerImpl()

    generator = ToyPrebuiltFeaturesGenerator(features)
    selector = ToyAllFeatureSelector()
    params = RegressionFeatureSearchParams(
        generators=(generator,),
        selector=selector
    )

    result = await feature_search_runner.run(run_context, input_data, params)
    expected_features = deduplicate_feature_names(features, [input_data.target_column])
    assert result.features == tuple(expected_features)

async def test_feature_has_same_name_as_input_column(input_data: FeatureSearchInputData, run_context: RunContext) -> None:
    await _test_feature_name_collision(input_data, run_context, (
        ToySyncFeature('x', 'y', 'x', '', ''),
        ToySyncFeature('x', 'y', 'y', '', '')
    ))

async def test_feature_has_same_name_as_target_column(input_data: FeatureSearchInputData, run_context: RunContext) -> None:
    await _test_feature_name_collision(input_data, run_context, (
        ToySyncFeature('x', 'y', 'x+y', '', ''),
        ToySyncFeature('x', 'y', input_data.target_column, '', '')
    ))

async def test_two_features_have_the_same_name(input_data: FeatureSearchInputData, run_context: RunContext) -> None:
    await _test_feature_name_collision(input_data, run_context, (
        ToySyncFeature('x', 'y', 'x+y', '', ''),
        ToySyncFeature('x', 'y', 'x+y', '', '')
    ))

def test_update_feature_defaults() -> None:
    runner: FeatureSearchRunnerImpl[Regression] = FeatureSearchRunnerImpl()

    float_feature = SimpleFloatFeature('f1', default_for_missing=math.inf, default_for_nan=math.inf,
                                       default_for_infinity=math.inf, default_for_neg_infinity=math.inf)

    adjusted_float_feature = runner._update_feature_defaults(float_feature, pl.Series([-1.0, 0.0, 1.0]))
    assert adjusted_float_feature.default_for_missing == 0.0
    assert adjusted_float_feature.default_for_nan == 0.0
    assert adjusted_float_feature.default_for_infinity == 2.0
    assert adjusted_float_feature.default_for_neg_infinity == -2.0
    assert type(adjusted_float_feature) is type(float_feature)

    bool_feature = SimpleBoolFeature('b1', default_for_missing=True)
    adjusted_bool_feature = runner._update_feature_defaults(bool_feature, pl.Series([True, False, True]))
    assert adjusted_bool_feature.default_for_missing is False
    assert type(adjusted_bool_feature) is type(bool_feature)

    int_feature = SimpleIntFeature('i1', default_for_missing=-1)
    adjusted_int_feature = runner._update_feature_defaults(int_feature, pl.Series([1, 2, 3]))
    assert adjusted_int_feature.default_for_missing == 2
    assert type(adjusted_int_feature) is type(int_feature)

    categorical_feature = SimpleCategoricalFeature('c1', default_for_missing='b')
    adjusted_categorical_feature = runner._update_feature_defaults(categorical_feature, pl.Series(['a', 'b', 'c']))
    assert adjusted_categorical_feature.default_for_missing == CategoricalFeature.other_category
    assert type(adjusted_categorical_feature) is type(categorical_feature)

async def test_correct_features_have_defaults_updated(input_data: FeatureSearchInputData, run_context: RunContext) -> None:
    runner: FeatureSearchRunnerImpl[Regression] = FeatureSearchRunnerImpl()

    feature = ToySyncFeature('x', 'y', 'x+y', '', '')
    generator = SimplePrebuiltFeaturesGenerator((GeneratedFeature(feature, False), GeneratedFeature(feature, True)))
    selector = ToyAllFeatureSelector()
    params = RegressionFeatureSearchParams(
        generators=(generator,),
        selector=selector
    )
    results = await runner.run(run_context, input_data, params)

    expected_features = deduplicate_feature_names([
        runner._update_feature_defaults(feature, input_data.feature_search.data.select((pl.col('x') + pl.col('y')).alias('x+y'))['x+y']),
        feature
    ])
    assert results.features == tuple(expected_features)


