import logging
from pathlib import Path

import polars as pl
import pytest
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

from agentune.analyze.context.base import TablesWithContextDefinitions
from agentune.analyze.core.database import DuckdbManager, DuckdbName, DuckdbTable
from agentune.analyze.core.dataset import DatasetSource
from agentune.analyze.core.duckdbio import (
    DuckdbTableSink,
)
from agentune.analyze.feature.base import Classification, Feature, Regression
from agentune.analyze.feature.dedup_names import deduplicate_feature_names
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
