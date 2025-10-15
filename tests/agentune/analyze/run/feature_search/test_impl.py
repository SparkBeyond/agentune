import logging
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, override

import attrs
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

from agentune.analyze.core.database import DuckdbManager, DuckdbName, DuckdbTable
from agentune.analyze.core.dataset import Dataset, DatasetSource
from agentune.analyze.core.duckdbio import (
    DuckdbTableSink,
)
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.feature.base import (
    BoolFeature,
    CategoricalFeature,
    Feature,
    FloatFeature,
    IntFeature,
)
from agentune.analyze.feature.dedup_names import deduplicate_feature_names
from agentune.analyze.feature.gen.base import GeneratedFeature, SyncFeatureGenerator
from agentune.analyze.feature.problem import Problem, ProblemDescription
from agentune.analyze.feature.select.base import (
    EnrichedFeatureSelector,
    FeatureSelector,
)
from agentune.analyze.join.base import JoinStrategy, TablesWithJoinStrategies
from agentune.analyze.run.base import RunContext
from agentune.analyze.run.feature_search.base import (
    FeatureSearchInputData,
    FeatureSearchParams,
    NoFeaturesFoundError,
)
from agentune.analyze.run.feature_search.impl import FeatureSearchRunnerImpl
from agentune.analyze.run.ingest import sampling

_logger = logging.getLogger(__name__)


@frozen
class TestFloatFeature(FloatFeature):
    name: str
    description: str = ''
    technical_description: str = ''
    default_for_missing: float = 0.0
    default_for_nan: float = 0.0
    default_for_infinity: float = 0.0
    default_for_neg_infinity: float = 0.0

    params: Schema = Schema(())
    secondary_tables: tuple[DuckdbTable, ...] = ()
    join_strategies: tuple[JoinStrategy, ...] = ()

    @override
    async def aevaluate(self, args: tuple[Any, ...], 
                        conn: DuckDBPyConnection) -> float | None:
        return 0.0

@frozen
class TestIntFeature(IntFeature):
    name: str
    description: str = ''
    technical_description: str = ''
    default_for_missing: int = 0

    params: Schema = Schema(())
    secondary_tables: tuple[DuckdbTable, ...] = ()
    join_strategies: tuple[JoinStrategy, ...] = ()

    @override
    async def aevaluate(self, args: tuple[Any, ...], 
                        conn: DuckDBPyConnection) -> int | None:
        return 0


@frozen
class TestBoolFeature(BoolFeature):
    name: str
    description: str = ''
    technical_description: str = ''
    default_for_missing: bool = True

    params: Schema = Schema(())
    secondary_tables: tuple[DuckdbTable, ...] = ()
    join_strategies: tuple[JoinStrategy, ...] = ()

    @override
    async def aevaluate(self, args: tuple[Any, ...], 
                        conn: DuckDBPyConnection) -> bool | None:
        return True

@frozen
class TestCategoricalFeature(CategoricalFeature):
    name: str
    description: str = ''
    technical_description: str = ''
    default_for_missing: str = CategoricalFeature.other_category

    categories: tuple[str, ...] = ('a', 'b', 'c')

    params: Schema = Schema(())
    secondary_tables: tuple[DuckdbTable, ...] = ()
    join_strategies: tuple[JoinStrategy, ...] = ()

    @override
    async def aevaluate(self, args: tuple[Any, ...], 
                        conn: DuckDBPyConnection) -> str | None:
        return None

@frozen
class SimplePrebuiltFeaturesGenerator(SyncFeatureGenerator):
    features: tuple[GeneratedFeature, ...]

    @override
    def generate(self, feature_search: Dataset, problem: Problem, join_strategies: TablesWithJoinStrategies,
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
        input_data = FeatureSearchInputData.from_split_table(split, TablesWithJoinStrategies.from_list([]), conn)
        return input_data

async def _test_feature_search(input_data: FeatureSearchInputData, run_context: RunContext,
                               params: FeatureSearchParams) -> None:
    # Limit the batch size to make sure multiple batches are tested
    feature_search_runner = FeatureSearchRunnerImpl(max_features_enrich_batch_size=5)

    with run_context.ddb_manager.cursor() as conn:
        orig_table_names = conn.execute('select database_name, schema_name, table_name from duckdb_tables()').fetchall()

    results = await feature_search_runner.run(run_context, input_data, params)
    assert results.enriched_train is None
    assert results.enriched_test is None

    # Regression test - check that all components are reusable and don't keep state from the last run
    # Don't require the order of the features to stay the same
    results2 = await feature_search_runner.run(run_context, input_data, params)
    assert set(results2.features_with_train_stats) == set(results.features_with_train_stats)
    assert set(results2.features_with_test_stats) == set(results.features_with_test_stats)
    assert attrs.evolve(results2, features_with_train_stats=results.features_with_train_stats,
                        features_with_test_stats=results.features_with_test_stats) == results

    with run_context.ddb_manager.cursor() as conn:
        table_names = conn.execute('select database_name, schema_name, table_name from duckdb_tables()').fetchall()
        assert set(orig_table_names) == set(table_names), 'No new tables left in DB (temporary or otherwise)'


async def test_feature_search(input_data: FeatureSearchInputData, run_context: RunContext) -> None:
    selectors: list[FeatureSelector | EnrichedFeatureSelector] = [ToySyncFeatureSelector(), ToyAsyncFeatureSelector(),
                                                                  ToySyncEnrichedFeatureSelector(), ToyAsyncEnrichedFeatureSelector()]
    # Repeat the same generators so that identical features are generated and need to be deduplicated
    # The features generated by the sync and async generators aren't identical even when they're summing the same columns,
    # because instances of ToySyncFeature and ToyAsyncFeature can't equal each other.
    generators = (ToySyncFeatureGenerator(), ToyAsyncFeatureGenerator(), ToySyncFeatureGenerator(), ToyAsyncFeatureGenerator())
    for selector in selectors:
        _logger.info(f'Running regression with selector {selector}')
        regression_params = FeatureSearchParams(
            ProblemDescription('target', 'regression'),
            generators=generators,
            selector=selector
        )
        await _test_feature_search(input_data, run_context, regression_params)
        _logger.info(f'Running classification with selector {selector}')
        classification_params = FeatureSearchParams(
            ProblemDescription('target', 'classification'),
            generators=generators,
            selector=selector
        )
        await _test_feature_search(input_data, run_context, classification_params)

async def _test_feature_name_collision(input_data: FeatureSearchInputData, run_context: RunContext,
                                       features: tuple[Feature, ...]) -> None:
    feature_search_runner = FeatureSearchRunnerImpl()

    generator = ToyPrebuiltFeaturesGenerator(features)
    selector = ToyAllFeatureSelector()
    params = FeatureSearchParams(
        ProblemDescription('target', 'regression'),
        generators=(generator,),
        selector=selector
    )

    result = await feature_search_runner.run(run_context, input_data, params)
    expected_features = deduplicate_feature_names(features, ['target'])
    assert result.features == tuple(expected_features)

async def test_feature_has_same_name_as_input_column(input_data: FeatureSearchInputData, run_context: RunContext) -> None:
    await _test_feature_name_collision(input_data, run_context, (
        ToySyncFeature('x', 'y', 'x', '', ''),
        ToySyncFeature('x', 'y', 'y', '', '')
    ))

async def test_feature_has_same_name_as_target_column(input_data: FeatureSearchInputData, run_context: RunContext) -> None:
    await _test_feature_name_collision(input_data, run_context, (
        ToySyncFeature('x', 'y', 'x+y', '', ''),
        ToySyncFeature('x', 'y', 'target', '', '')
    ))

async def test_two_features_have_the_same_name(input_data: FeatureSearchInputData, run_context: RunContext) -> None:
    await _test_feature_name_collision(input_data, run_context, (
        ToySyncFeature('x', 'y', 'x+y', '', ''),
        ToySyncFeature('x', 'y', 'x+y', '', '')
    ))

def test_update_feature_defaults() -> None:
    runner = FeatureSearchRunnerImpl()

    float_feature = TestFloatFeature('f1', default_for_missing=math.inf, default_for_nan=math.inf,
                                     default_for_infinity=math.inf, default_for_neg_infinity=math.inf)

    adjusted_float_feature = runner._update_feature_defaults(float_feature, pl.Series([-1.0, 0.0, 1.0]))
    assert isinstance(adjusted_float_feature, FloatFeature)
    assert adjusted_float_feature.default_for_missing == 0.0
    assert adjusted_float_feature.default_for_nan == 0.0
    assert adjusted_float_feature.default_for_infinity == 2.0
    assert adjusted_float_feature.default_for_neg_infinity == -2.0
    assert type(adjusted_float_feature) is type(float_feature)

    bool_feature = TestBoolFeature('b1', default_for_missing=True)
    adjusted_bool_feature = runner._update_feature_defaults(bool_feature, pl.Series([True, False, True]))
    assert adjusted_bool_feature.default_for_missing is False
    assert type(adjusted_bool_feature) is type(bool_feature)

    int_feature = TestIntFeature('i1', default_for_missing=-1)
    adjusted_int_feature = runner._update_feature_defaults(int_feature, pl.Series([1, 2, 3]))
    assert adjusted_int_feature.default_for_missing == 2
    assert type(adjusted_int_feature) is type(int_feature)

    categorical_feature = TestCategoricalFeature('c1', default_for_missing='b')
    adjusted_categorical_feature = runner._update_feature_defaults(categorical_feature, pl.Series(['a', 'b', 'c']))
    assert adjusted_categorical_feature.default_for_missing == CategoricalFeature.other_category
    assert type(adjusted_categorical_feature) is type(categorical_feature)

async def test_correct_features_have_defaults_updated(input_data: FeatureSearchInputData, run_context: RunContext) -> None:
    runner = FeatureSearchRunnerImpl()

    feature = ToySyncFeature('x', 'y', 'x+y', '', '')
    generator = SimplePrebuiltFeaturesGenerator((GeneratedFeature(feature, False), GeneratedFeature(feature, True)))
    selector = ToyAllFeatureSelector()
    params = FeatureSearchParams(
        ProblemDescription('target', 'regression'),
        generators=(generator,),
        selector=selector
    )
    results = await runner.run(run_context, input_data, params)

    expected_features = deduplicate_feature_names([
        runner._update_feature_defaults(feature, input_data.feature_search.data.select((pl.col('x') + pl.col('y')).alias('x+y'))['x+y']),
        feature
    ])
    assert results.features == tuple(expected_features)


def input_data_from_df(run_context: RunContext, df: pl.DataFrame) -> FeatureSearchInputData:
    with run_context.ddb_manager.cursor() as conn:
        conn.register('df', df)
        conn.execute('create or replace table input as from df')
        table = DuckdbTable.from_duckdb('input', conn)
        split = sampling.split_duckdb_table(conn, table.name)
        return FeatureSearchInputData.from_split_table(split, TablesWithJoinStrategies.from_list([]), conn)

async def test_keep_enrich_output(input_data: FeatureSearchInputData, run_context: RunContext) -> None:
    with run_context.ddb_manager.cursor() as conn:
        runner = FeatureSearchRunnerImpl()

        enriched_train_name = DuckdbName.qualify('enriched_train', conn)
        enriched_test_name = DuckdbName.qualify('enriched_test', conn)

        orig_table_names = conn.execute('select database_name, schema_name, table_name from duckdb_tables()').fetchall()

        feature = ToySyncFeature('x', 'y', 'x+y', '', '')
        generator = SimplePrebuiltFeaturesGenerator((GeneratedFeature(feature, False), GeneratedFeature(feature, True)))
        selector = ToyAllFeatureSelector()
        params = FeatureSearchParams(
            ProblemDescription('target', 'regression'),
            generators=(generator,),
            selector=selector,
            store_enriched_train=enriched_train_name,
            store_enriched_test=enriched_test_name
        )
        results = await runner.run(run_context, input_data, params)

        assert results.enriched_train == DuckdbTable.from_duckdb(enriched_train_name, conn), 'Enriched train was stored'
        assert results.enriched_test == DuckdbTable.from_duckdb(enriched_test_name, conn), 'Enriched test was stored'

        assert results.enriched_train.schema == results.enriched_test.schema, 'Same schema for enriched train and test'
        assert results.enriched_train.schema == Schema((
            input_data.feature_search.schema['target'],
            * [Field(feature.name, feature.dtype) for feature in results.features]
        )), 'Enriched train contains the target column and a column for each feature'

        enriched_train_len = conn.execute(f'select count(*) from {enriched_train_name}').fetchone()
        assert enriched_train_len == (input_data.feature_eval.to_dataset(conn).height, ), 'Enriched train has expected number of rows'
        enriched_test_len = conn.execute(f'select count(*) from {enriched_test_name}').fetchone()
        assert enriched_test_len == (input_data.test.to_dataset(conn).height, ), 'Enriched test has expected number of rows'

        table_names = conn.execute('select database_name, schema_name, table_name from duckdb_tables()').fetchall()
        expected_table_names = [
            *orig_table_names,
            (enriched_train_name.database, enriched_train_name.schema, enriched_train_name.name),
            (enriched_test_name.database, enriched_test_name.schema, enriched_test_name.name),
        ]
        assert set(table_names) == set(expected_table_names), 'No other new tables left in DB (temporary or otherwise)'

        # Run again, should overwrite previous contents of enriched tables if they exist
        # Change the tables' contents first to notice that they change back
        conn.execute(f'insert into {enriched_train_name} select * from {enriched_test_name}')
        conn.execute(f'insert into {enriched_test_name} select * from {enriched_train_name}')

        results2 = await runner.run(run_context, input_data, params)
        assert results2 == results, 'Same features selected, same stats calculated'

        enriched_train_len = conn.execute(f'select count(*) from {enriched_train_name}').fetchone()
        assert enriched_train_len == (input_data.feature_eval.to_dataset(conn).height, ), 'Enriched train has expected number of rows'
        enriched_test_len = conn.execute(f'select count(*) from {enriched_test_name}').fetchone()
        assert enriched_test_len == (input_data.test.to_dataset(conn).height, ), 'Enriched test has expected number of rows'

async def test_zero_generated_features(input_data: FeatureSearchInputData, run_context: RunContext) -> None:
    runner = FeatureSearchRunnerImpl()
    generator = SimplePrebuiltFeaturesGenerator(())
    selectors: list[FeatureSelector | EnrichedFeatureSelector] = [ToyAllFeatureSelector(), ToySyncEnrichedFeatureSelector()]
    for selector in selectors:
        params = FeatureSearchParams(
            ProblemDescription('target', 'regression'),
            generators=(generator,),
            selector=selector
        )
        with pytest.raises(NoFeaturesFoundError):
            await runner.run(run_context, input_data, params)

