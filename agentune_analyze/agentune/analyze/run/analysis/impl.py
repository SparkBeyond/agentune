import asyncio
import itertools
import logging
import math
from collections.abc import Sequence
from typing import cast, override

import attrs
import polars as pl
from attrs import frozen
from duckdb import DuckDBPyConnection, DuckDBPyRelation

from agentune.analyze.core import default_duckdb_batch_size
from agentune.analyze.core.database import (
    DuckdbManager,
    DuckdbName,
    DuckdbTable,
)
from agentune.analyze.core.dataset import DatasetSink, DatasetSource
from agentune.analyze.core.schema import restore_df_types
from agentune.analyze.feature.base import (
    BoolFeature,
    CategoricalFeature,
    Feature,
    FloatFeature,
    IntFeature,
)
from agentune.analyze.feature.dedup_names import deduplicate_feature_names, deduplicate_strings
from agentune.analyze.feature.gen.base import (
    FeatureGenerator,
    GeneratedFeature,
    SyncFeatureGenerator,
)
from agentune.analyze.feature.problem import Problem, ProblemDescription
from agentune.analyze.feature.select.base import (
    FeatureSelector,
    SyncEnrichedFeatureSelector,
    SyncFeatureSelector,
)
from agentune.analyze.feature.stats import stats_calculators
from agentune.analyze.feature.stats.base import FeatureWithFullStats, FullFeatureStats
from agentune.analyze.run.analysis import problem_discovery
from agentune.analyze.run.analysis.base import (
    AnalyzeComponents,
    AnalyzeInputData,
    AnalyzeParams,
    AnalyzeResults,
    AnalyzeRunner,
    NoFeaturesFoundError,
    UniqueTableName,
)
from agentune.analyze.util.queue import Queue, ScopedQueue

_logger = logging.getLogger(__name__)

@frozen
class AnalyzeRunnerImpl(AnalyzeRunner):
    """The analysis process consists of:

    - Generate candidate features (from all generators)
    - Enrich and calculate stats on feature_search dataset
    - Select final features, and deduplicate their names
    - Enrich and calculate stats on feature_eval and test datasets, and return those statistics

    The enriched data (and any other temporary data stored in duckdb) is stored in the dedicated temp schema
    of the main database. This avoids name conflicts and ensures it's eventually discarded when the RunContext
    and therefore the duckdb database are closed.

    Current limitations:
    - All generated features are kept in memory at once. (A future version could offload them
      by serializing them, potentially into duckdb.)
    - Not all features are enriched at once (for fear of resource limits); the grouping is naive and,
      with more than one feature generator running in parallel, will likely be suboptimal.
    - The enriched feature_search dataset is discarded before enriching the feature_eval dataset,
      although they may share rows that could be reused

    Args:
        max_features_enrich_batch_size: Enrich at most these many features at once.
                                        If there are more candidate features, enrich them in batches of this size.
        run_generators_concurrently:    If True, all supplied async FeatureGenerators run at once, as well as
                                        up to one SyncFeatureGenerator at a time.
    """

    max_features_enrich_batch_size: int = 1000
    run_generators_concurrently: bool = True
    batch_size: int = default_duckdb_batch_size

    @override
    async def run(self, ddb_manager: DuckdbManager, data: AnalyzeInputData,
                  params: AnalyzeParams, components: AnalyzeComponents,
                  problem_description: ProblemDescription) -> AnalyzeResults:

        with ddb_manager.cursor() as conn:
            problem = await asyncio.to_thread(problem_discovery.discover_problem, data.copy_to_thread(),
                                              problem_description, conn, params.max_classes)
            await asyncio.to_thread(problem_discovery.validate_input, data.copy_to_thread(), problem, conn)

        with ddb_manager.cursor() as conn:
            candidate_features = await self._generate_features(conn, data, components.generators, problem)

        if len(candidate_features) == 0:
            raise NoFeaturesFoundError

        # Later we will go back to the original list to recover the original name of each selected feature,
        # since after selection not all deduplication will be needed
        deduplicated_candidate_features = self._deduplicate_generated_feature_names(candidate_features, existing_names=[problem.target_column.name])

        # Compute candidate features on the feature_search dataset, storing the results in the temp schema.
        (enriched_feature_eval_group_tables, features_with_updated_defaults) = await self._enrich_in_batches_and_update_defaults(
            deduplicated_candidate_features, data.feature_eval,
            ddb_manager, components, 'enriched_feature_search',
            problem.target_column.name
        )
        try:
            with ddb_manager.cursor() as conn:
                selected_features = await self._select_features(features_with_updated_defaults,
                                                                data.feature_eval, enriched_feature_eval_group_tables,
                                                                problem, params, components, conn)
                _logger.debug(f'Selected {len(selected_features)} features out of {len(features_with_updated_defaults)}')
        finally:
            with ddb_manager.cursor() as conn:
                for table in enriched_feature_eval_group_tables:
                    conn.execute(f'drop table {table.name}')

        # Get the original version of the selected features, and re-deduplicate their names
        # Some of the original deduplications may no longer be necessary.
        # Note that we need to use the selected feature and not the original feature at that index,
        # because it has updated defaults.
        def original_name(feature: Feature) -> str:
            index = next(idx for idx, gen in enumerate(deduplicated_candidate_features) if gen.feature.name == feature.name)
            return candidate_features[index].feature.name

        selected_features_with_original_names = [attrs.evolve(feature, name=original_name(feature)) for feature in selected_features]
        deduplicated_selected_features = deduplicate_feature_names(selected_features_with_original_names, existing_names=[problem.target_column.name])

        enriched_eval_name = self._enriched_table_name(ddb_manager, params.store_enriched_train, 'enriched_eval')
        enriched_test_name = self._enriched_table_name(ddb_manager, params.store_enriched_test, 'enriched_test')

        try:
            with ddb_manager.cursor() as conn:
                enriched_eval_sink = DatasetSink.into_duckdb_table(enriched_eval_name)
                await components.enrich_runner.run_stream(deduplicated_selected_features, data.feature_eval,
                                                      enriched_eval_sink, components.feature_computers, conn,
                                                      keep_input_columns=(problem.target_column.name,),
                                                      deduplicate_names=False)
                features_with_eval_stats: list[FeatureWithFullStats] = \
                    await self._calculate_feature_stats_single_data(deduplicated_selected_features,
                                                                    enriched_eval_sink.as_source(conn),
                                                                    problem, conn)
                if not params.store_enriched_train:
                    conn.execute(f'drop table {enriched_eval_name}')

                enriched_test_sink = DatasetSink.into_duckdb_table(enriched_test_name)
                await components.enrich_runner.run_stream(deduplicated_selected_features, data.test,
                                                      enriched_test_sink, components.feature_computers, conn,
                                                      keep_input_columns=(problem.target_column.name,),
                                                      deduplicate_names=False)
                features_with_test_stats: list[FeatureWithFullStats] = \
                    await self._calculate_feature_stats_single_data(deduplicated_selected_features,
                                                                    enriched_test_sink.as_source(conn),
                                                                    problem, conn)
                if not params.store_enriched_test:
                    conn.execute(f'drop table {enriched_test_name}')

                return AnalyzeResults(problem,
                                      tuple(features_with_eval_stats), tuple(features_with_test_stats),
                                      DuckdbTable.from_duckdb(enriched_eval_name, conn) if params.store_enriched_train else None,
                                      DuckdbTable.from_duckdb(enriched_test_name, conn) if params.store_enriched_test else None)
        finally:
            with ddb_manager.cursor() as conn:
                if not params.store_enriched_train:
                    conn.execute(f'drop table if exists {enriched_eval_name}')
                if not params.store_enriched_test:
                    conn.execute(f'drop table if exists {enriched_test_name}')

    def _enriched_table_name(self, ddb_manager: DuckdbManager,
                             user_requested: str | DuckdbName | UniqueTableName | None,
                             default_basename: str) -> DuckdbName:
        with ddb_manager.cursor() as conn:
            match user_requested:
                case str() as s: return DuckdbName.qualify(s, conn)
                case DuckdbName() as n: return n
                case UniqueTableName(basename):
                    match basename:
                        case str(): return DuckdbName.qualify(ddb_manager.random_name(basename), conn)
                        case DuckdbName(): return attrs.evolve(basename, name=ddb_manager.random_name(basename.name))
                case None: return ddb_manager.temp_random_name(default_basename)

    def _deduplicate_generated_feature_names(self, features: Sequence[GeneratedFeature],
                                             existing_names: Sequence[str] = ()) -> list[GeneratedFeature]:
        return [GeneratedFeature(attrs.evolve(gen.feature, name=new_name), gen.has_good_defaults)
                if new_name != gen.feature.name else gen
                for gen, new_name in zip(features, deduplicate_strings([gen.feature.name for gen in features],
                                                                       existing=existing_names), strict=False)]

    async def _generate_features(self, conn: DuckDBPyConnection, data: AnalyzeInputData,
                                 generators: Sequence[FeatureGenerator], problem: Problem) -> list[GeneratedFeature]:
        async with ScopedQueue[GeneratedFeature](maxsize=0) as queue: # maxsize=0 means unlimited
            sync_generators = [generator for generator in generators if isinstance(generator, SyncFeatureGenerator)]
            async_generators = [generator for generator in generators if not isinstance(generator, SyncFeatureGenerator)]

            if self.run_generators_concurrently:
                await asyncio.gather(
                    self._generate_sync(conn, queue, data, sync_generators, problem),
                    self._generate_async(conn, queue, data, async_generators, problem)
                )
            else:
                await self._generate_sync(conn, queue, data, sync_generators, problem)
                await self._generate_async(conn, queue, data, async_generators, problem)

            queue.close() # so that iteration will terminate when producing the list()
            return list(queue)

    async def _generate_sync(self, conn: DuckDBPyConnection, output_queue: Queue[GeneratedFeature], data: AnalyzeInputData,
                             generators: list[SyncFeatureGenerator], problem: Problem) -> None:
        if not generators:
            return

        with conn.cursor() as cursor: # Cursor for new thread
            def sync_generate() -> None:
                for generator in generators:
                    _logger.debug(f'Generating features with {generator=}')
                    count = 0
                    for feature in generator.generate(data.feature_search, problem, data.join_strategies, cursor):
                        output_queue.put(feature)
                        count += 1
                    _logger.debug(f'Generated {count} features with {generator=}')
            await asyncio.to_thread(sync_generate)

    async def _generate_async(self, conn: DuckDBPyConnection, output_queue: Queue[GeneratedFeature], data: AnalyzeInputData,
                              generators: list[FeatureGenerator], problem: Problem) -> None:
        async def agenerate(generator: FeatureGenerator) -> None:
            _logger.debug(f'Generating features with {generator=}')
            count = 0
            async for feature in generator.agenerate(data.feature_search, problem, data.join_strategies, conn):
                await output_queue.aput(feature)
                count += 1
            _logger.debug(f'Generated {count} features with {generator=}')

        if self.run_generators_concurrently:
            await asyncio.gather(*[agenerate(generator) for generator in generators])
        else:
            for generator in generators:
                await agenerate(generator)

    async def _enrich_in_batches_and_update_defaults(self, features: list[GeneratedFeature], dataset_source: DatasetSource,
                                                     ddb_manager: DuckdbManager,
                                                     components: AnalyzeComponents, target_table_base_name: str,
                                                     target_column: str) -> tuple[list[DuckdbTable], list[Feature]]:
        """Enrich these features in batches of size up to self.max_features_enrich_batch_size,
        and return a table per batch. The first table also has the target column; the rest don't.

        Also, update the default values of features with has_good_defaults=False. Return a list of all input features
        with the final default values.
        """
        feature_groups = list(itertools.batched(features, self.max_features_enrich_batch_size))
        tables = []
        features_with_updated_defaults = []
        try:
            with ddb_manager.cursor() as conn:
                for index, feature_group in enumerate(feature_groups):
                    keep_input_columns = (target_column,) if index == 0 else ()
                    group_table_name = ddb_manager.temp_random_name(target_table_base_name)
                    dataset_sink = DatasetSink.into_duckdb_table(group_table_name)
                    await components.enrich_runner.run_stream([gen.feature for gen in feature_group],
                                                              dataset_source, dataset_sink, components.feature_computers,
                                                              conn, keep_input_columns=keep_input_columns,
                                                              deduplicate_names=False)
                    table = DuckdbTable.from_duckdb(group_table_name, conn)
                    tables.append(table)

                    for gen in feature_group:
                        if gen.has_good_defaults:
                            features_with_updated_defaults.append(gen.feature)
                        else:
                            rel = conn.table(str(table.name)).select(f'"{gen.feature.name}"')
                            df = restore_df_types(rel.pl(), table.schema.select(gen.feature.name))
                            series = df[gen.feature.name]
                            features_with_updated_defaults.append(self._update_feature_defaults(gen.feature, series))

                _logger.debug(f'Enriched {len(features)} features in {len(feature_groups)} batches')
                return (tables, features_with_updated_defaults)
        except:
            # On error, delete tables created so far before returning
            with ddb_manager.cursor() as conn:
                for table in tables:
                    conn.execute(f'drop table {table.name}')
            raise

    def _join_tables(self, tables: list[DuckdbTable], conn: DuckDBPyConnection) -> DatasetSource:
        """Join several tables on their 'default' order (i.e. the rowid).

        This is not well-defined in SQL, but it is in duckdb.
        """
        if len(tables) == 1:
            return DatasetSource.from_table(tables[0], self.batch_size)

        join_clauses = [f'JOIN {table.name} AS "{table.name.name}" ON "{tables[0].name.name}".rowid = "{table.name.name}".rowid' for table in tables[1:]]
        query = f'''SELECT {', '.join(f'"{table.name.name}".*' for table in tables)} 
                    FROM {tables[0].name} AS "{tables[0].name.name}"
                    {'\n'.join(join_clauses)}
                  '''

        def read(conn: DuckDBPyConnection) -> DuckDBPyRelation:
            return conn.sql(query)

        return DatasetSource.from_duckdb_parser(read, conn, self.batch_size)

    async def _select_features(self, candidate_features: list[Feature], feature_eval: DatasetSource,
                               enriched_groups: list[DuckdbTable], problem: Problem,
                               params: AnalyzeParams, components: AnalyzeComponents,
                               conn: DuckDBPyConnection) -> list[Feature]:
        selector = components.selector
        if isinstance(selector, FeatureSelector):
            target_series = feature_eval.select(problem.target_column.name).to_dataset(conn).data[problem.target_column.name]
            features_with_data: list[tuple[Feature, DatasetSource]] = [
                (feature, DatasetSource.from_table(next(table for table in enriched_groups if feature.name in table.schema.names)))
                for feature in candidate_features
            ]
            features_with_stats: list[FeatureWithFullStats] = \
                await self._calculate_feature_stats(features_with_data, target_series, problem, conn)

            if isinstance(selector, SyncFeatureSelector):
                def sync_select() -> list[Feature]:
                    for fws in features_with_stats:
                        selector.add_feature(fws)
                    return [fws.feature for fws in selector.select_final_features(problem, params.max_features_to_select)]

                return await asyncio.to_thread(sync_select)

            else:
                for fws in features_with_stats:
                    await selector.aadd_feature(fws)
                return [fws.feature for fws in await selector.aselect_final_features(problem, params.max_features_to_select)]

        else:
            # selector is EnrichedFeatureSelector. The first enriched table also contains the target column.
            enriched_source = self._join_tables(enriched_groups, conn)

            if isinstance(selector, SyncEnrichedFeatureSelector):
                with conn.cursor() as cursor: # for new thread
                    def select() -> list[Feature]:
                        return list(selector.select_features(candidate_features, params.max_features_to_select, enriched_source, problem, cursor))
                    return await asyncio.to_thread(select)

            else:
                return list(await selector.aselect_features(candidate_features, params.max_features_to_select, enriched_source, problem, conn))

    def _update_feature_defaults(self, feature: Feature, enriched: pl.Series) -> Feature:
        match feature:
            case IntFeature():
                return attrs.evolve(feature, default_for_missing=int(cast(float, enriched.median())))
            case BoolFeature():
                return attrs.evolve(feature, default_for_missing=False)
            case CategoricalFeature():
                return attrs.evolve(feature, default_for_missing=CategoricalFeature.other_category)
            case FloatFeature():
                finite_values = enriched.replace([math.inf, -math.inf], [None, None])
                max_val = cast(float, finite_values.max()) + 1
                min_val = cast(float, finite_values.min()) - 1
                substituted = enriched.replace([math.inf, -math.inf], [max_val, min_val])
                median = cast(float, substituted.median())
                return attrs.evolve(feature, default_for_missing=median, default_for_nan=median,
                                    default_for_infinity=max_val, default_for_neg_infinity=min_val)
            case _:
                raise TypeError(f'Unexpected feature type {type(feature)}')


    async def _calculate_feature_stats(self, features_with_data: list[tuple[Feature, DatasetSource]],
                                       target_series: pl.Series, problem: Problem,
                                       conn: DuckDBPyConnection) -> list[FeatureWithFullStats]:
        # Stats calculators are always synchronous. We run them on a single thread, one feature at a time;
        # we could use several threads.
        with conn.cursor() as cursor: # for new thread
            def calculate() -> list[FeatureWithFullStats]:
                result = []
                for feature, data_source in features_with_data:
                    dataset = data_source.select(feature.name).to_dataset(cursor)
                    feature_stats_calculator = stats_calculators.get_feature_stats_calculator(feature, problem)
                    feature_stats = feature_stats_calculator.calculate_from_series(feature, dataset.data[feature.name])
                    relationship_stats_calculator = stats_calculators.get_relationship_stats_calculator(feature, problem)
                    relationship_stats = relationship_stats_calculator.calculate_from_series(feature, dataset.data[feature.name],
                                                                                             target_series, problem)
                    feature_with_stats = FeatureWithFullStats(feature, FullFeatureStats(feature_stats, relationship_stats))
                    result.append(feature_with_stats)
                return result

            return await asyncio.to_thread(calculate)

    async def _calculate_feature_stats_single_data(self, features: list[Feature],
                                                   dataset_source: DatasetSource, problem: Problem,
                                                   conn: DuckDBPyConnection) -> list[FeatureWithFullStats]:
        target_source = dataset_source.select(problem.target_column.name)
        with conn.cursor() as cursor:
            target_series = (await asyncio.to_thread(target_source.to_dataset, cursor)).data[problem.target_column.name]

        return await self._calculate_feature_stats([(feature, dataset_source) for feature in features], target_series, problem, conn)
