import logging
from pathlib import Path

import polars as pl
import pytest
from tests.agentune.analyze.run.analysis.toys import (
    ToyAsyncEnrichedFeatureSelector,
    ToyAsyncFeatureGenerator,
    ToySyncFeatureGenerator,
)

from agentune.analyze.feature.problem import ProblemDescription
from agentune.analyze.run.analysis.base import (
    AnalyzeComponents,
    NoFeaturesFoundError,
)
from agentune.api.base import RunContext
from agentune.api.data import BoundTable
from agentune.core.database import (
    DuckdbName,
    DuckdbOnDisk,
    DuckdbTable,
)
from agentune.core.dataset import Dataset, DatasetSource, ReadCsvParams

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

async def test_e2e_flow_synthetic(input_data_csv_path: Path, tmp_path: Path) -> None:
    async with await RunContext.create() as ctx:
        input = await ctx.data.from_csv(input_data_csv_path).copy_to_table('input')
        split_input = await input.split()
        problem_description = ProblemDescription('target', 'Test synthetic data problem')

        with pytest.raises(NoFeaturesFoundError): # Default generators can't do anything with this synthetic data
            await ctx.ops.analyze(problem_description, split_input)

        components = AnalyzeComponents(
            generators=(ToySyncFeatureGenerator(), ToyAsyncFeatureGenerator(), ToySyncFeatureGenerator(), ToyAsyncFeatureGenerator()),
            selector=ToyAsyncEnrichedFeatureSelector()
        )
        results = await ctx.ops.analyze(problem_description, split_input, components=components)

        # We can store & load results (including the features and stats) as JSON
        assert len(results.features) > 0
        results_json = ctx.json.dumps(results)
        results2 = ctx.json.loads(results_json, type(results))
        assert results == results2

        # Enrich new data (we don't have actually new data so we'll enrich a copy of the original data in a new location)
        new_input = await ctx.data.from_csv(input_data_csv_path).copy_to_table('new_input')
        output_path = tmp_path / 'output.csv'
        output = ctx.data.to_csv(output_path)
        await ctx.ops.enrich_stream(results.features, new_input, output)

        enriched = ctx.data.from_csv(output_path)
        assert await enriched.size() == await new_input.size()
        assert list(enriched.schema.names) == [feature.name for feature in results.features]

async def test_rename_main_database(input_data_csv_path: Path, tmp_path: Path) -> None:
    dbpath = tmp_path / 'duck.db'
    async with await RunContext.create(DuckdbOnDisk(dbpath)) as ctx:
        input = await ctx.data.from_csv(input_data_csv_path).copy_to_table('input')
        split_input = await input.split()
        problem_description = ProblemDescription('target', 'Test synthetic data problem')

        components = AnalyzeComponents(
            generators=(ToySyncFeatureGenerator(), ToyAsyncFeatureGenerator(), ToySyncFeatureGenerator(), ToyAsyncFeatureGenerator()),
            selector=ToyAsyncEnrichedFeatureSelector()
        )
        results = await ctx.ops.analyze(problem_description, split_input, components=components)
        results_json = ctx.json.dumps(results)

    dbpath2 = tmp_path / 'duck2.db'
    dbpath.rename(dbpath2)

    async with await RunContext.create(DuckdbOnDisk(dbpath2)) as ctx:
        results = ctx.json.loads(results_json, type(results))

        new_input = await ctx.data.from_csv(input_data_csv_path).copy_to_table('new_input')
        output_path = tmp_path / 'output.csv'
        output = ctx.data.to_csv(output_path)
        await ctx.ops.enrich_stream(results.features, new_input, output)

        enriched = ctx.data.from_csv(output_path)
        assert await enriched.size() == await new_input.size()
        assert list(enriched.schema.names) == [feature.name for feature in results.features]

async def test_analyze_separate_test_input(input_data_csv_path: Path) -> None:
    async with await RunContext.create() as ctx:
        input = await ctx.data.from_csv(input_data_csv_path).copy_to_table('input')
        split_input = await input.split()
        problem_description = ProblemDescription('target', 'Test synthetic data problem')
        components = AnalyzeComponents(
            generators=(ToySyncFeatureGenerator(),),
            selector=ToyAsyncEnrichedFeatureSelector()
        )

        results1 = await ctx.ops.analyze(problem_description, split_input, components=components)
        assert results1.features_with_test_stats[0].stats.feature.n_total == await split_input.test.size()

        custom_test = (await split_input.test.load()).data.limit(123)
        assert custom_test.height != await split_input.test.size()

        test_inputs: list[pl.DataFrame | Dataset | DatasetSource] = \
            [custom_test, Dataset.from_polars(custom_test), DatasetSource.from_dataset(Dataset.from_polars(custom_test))]
        for test_input in test_inputs:
            result = await ctx.ops.analyze(problem_description, split_input, test_input, components=components)
            assert result.features_with_test_stats[0].stats.feature.n_total == custom_test.height

        test_input2 = await ctx.data.from_df(test_input).copy_to_table('test_input2')
        test_inputs_tables: list[BoundTable | DuckdbTable | DuckdbName | str] = \
            [test_input2, test_input2.table, test_input2.table.name, test_input2.table.name.name]
        for test_input_table in test_inputs_tables:
            result = await ctx.ops.analyze(problem_description, split_input, test_input_table, components=components)
            assert result.features_with_test_stats[0].stats.feature.n_total == custom_test.height


@pytest.mark.integration
async def test_e2e_flow_real(test_data_conversations: dict[str, Path], tmp_path: Path) -> None:
    async with await RunContext.create() as ctx:
        main_path = test_data_conversations['main_csv']
        secondary_path = test_data_conversations['conversations_csv']

        # Analyze
        main = await ctx.data.from_csv(main_path).copy_to_table('input')
        split_input = await main.split()
        secondary = await ctx.data.from_csv(secondary_path, ReadCsvParams(dtype={'timestamp': 'timestamp'})).copy_to_table('secondary')
        join = secondary.join_strategy.conversation('conversations', 'id', 'id', 'timestamp', 'role', 'content')
        problem_description = ProblemDescription('outcome', 'Test conversation outcome prediction', target_desired_outcome='resolved')
        results = await ctx.ops.analyze(problem_description, split_input, secondary_tables=[secondary], join_strategies=[join])
        assert len(results.features) > 0

        # Action recommendations
        recommendations = await ctx.ops.recommend_conversation_actions(split_input, results)
        assert recommendations is not None

        # Enrich new data
        new_input = await ctx.data.from_csv(main_path).copy_to_table('new_input')
        output_path = tmp_path / 'output.csv'
        output = ctx.data.to_csv(output_path)
        await ctx.ops.enrich_stream(results.features, new_input, output)

        enriched = ctx.data.from_csv(output_path)
        assert await enriched.size() == await new_input.size()
        assert list(enriched.schema.names) == [feature.name for feature in results.features]
