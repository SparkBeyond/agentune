import asyncio
import itertools
import logging
import math
from collections.abc import Sequence
from typing import cast, override

import attrs
import polars as pl
from attrs import frozen
from duckdb.duckdb import DuckDBPyConnection, DuckDBPyRelation
from tests.agentune.analyze.core import default_duckdb_batch_size

from agentune.analyze.context.base import TablesWithContextDefinitions
from agentune.analyze.core.database import DuckdbInMemoryDatabase, DuckdbName, DuckdbTable
from agentune.analyze.core.dataset import Dataset, DatasetSink, DatasetSource
from agentune.analyze.core.schema import restore_df_types
from agentune.analyze.feature.base import (
    BoolFeature,
    CategoricalFeature,
    Feature,
    FloatFeature,
    IntFeature,
    TargetKind,
)
from agentune.analyze.feature.dedup_names import deduplicate_feature_names, deduplicate_strings
from agentune.analyze.feature.gen.base import (
    FeatureGenerator,
    GeneratedFeature,
    SyncFeatureGenerator,
)
from agentune.analyze.feature.select.base import (
    FeatureSelector,
    SyncEnrichedFeatureSelector,
    SyncFeatureSelector,
)
from agentune.analyze.feature.stats.base import FeatureWithFullStats, FullFeatureStats
from agentune.analyze.run.base import RunContext
from agentune.analyze.run.enrich.base import EnrichRunner
from agentune.analyze.run.enrich.impl import EnrichRunnerImpl
from agentune.analyze.run.feature_search.base import (
    FeatureSearchInputData,
    FeatureSearchParams,
    FeatureSearchResults,
    FeatureSearchRunner,
)
from agentune.analyze.util.queue import Queue, ScopedQueue

_logger = logging.getLogger(__name__)

@frozen
class FeatureSearchRunnerImpl[TK: TargetKind](FeatureSearchRunner[TK]):
    """The feature search process consists of:

    - Generate candidate features (from all generators)
    - Enrich and calculate stats on feature_search dataset
    - Select final features, and deduplicate their names
    - Enrich and calculate stats on feature_eval and test datasets, and return those statistics

    The enriched data (and any other temporary data stored in duckdb) is stored in a new in-memory database,
    to avoid conflicts with other code or data and to ensure it's closed and discarded when we're done.
    Like every in-memory database, it can spillover to disk if duckdb runs out of memory and spillover
    is enabled.

    Current limitations:
    - All generated features are kept in memory at once. (A future version could offload them
      by serializing them into duckdb.)
    - Not all features are enriched at once (for fear of resource limits); the grouping is naive and,
      with more than one feature generator running in parallel, will likely be suboptimal.
    - The enriched feature_search dataset is discarded before enriching the feature_eval dataset,
      although they may share rows that could be reused
    - The enriched feature_eval and test data are discarded, and only the feature statistics are returned
      (this is a limitation of the FeatureSearch API)

    Args:
        max_features_enrich_batch_size: Enrich at most these many features at once.
                                        If there are more candidate features, enrich them in batches of this size.
        run_generators_concurrently:    If True, all supplied async FeatureGenerators run at once, as well as
                                        up to one SyncFeatureGenerator at a time.
    """

    max_features_enrich_batch_size: int = 1000
    run_generators_concurrently: bool = True
    enrich_runner: EnrichRunner = EnrichRunnerImpl()
    batch_size: int = default_duckdb_batch_size

    @override
    async def run(self, run_context: RunContext, data: FeatureSearchInputData,
                  params: FeatureSearchParams[TK]) -> FeatureSearchResults[TK]:

        with run_context.ddb_manager.cursor() as conn:
            await asyncio.to_thread(self._validate_input, data.copy_to_thread(), conn)

        with run_context.ddb_manager.cursor() as conn:
            candidate_features = await self._generate_features(conn, data, params.generators)

        # Later we will go back to the original list to recover the original name of each selected feature,
        # since after selection not all deduplication will be needed
        deduplicated_candidate_features = self._deduplicate_generated_feature_names(candidate_features, existing_names=[data.target_column])

        # Evaluate candidate features on the feature_search dataset, storing the results in a temporary database.
        # We use a temp database not only because we discard it later, but because we're going to create
        # a dynamic number of tables in it.

        temp_db_name = 'feature_search_temp_db' # TODO nonce name
        run_context.ddb_manager.attach(DuckdbInMemoryDatabase(), temp_db_name)
        try:
            with run_context.ddb_manager.cursor() as conn:
                (enriched_feature_search_group_tables, features_with_updated_defaults) = await self._enrich_in_batches_and_update_defaults(
                    deduplicated_candidate_features, data.feature_search,
                    data.contexts, conn, params, DuckdbName('enriched_feature_search', temp_db_name),
                    data.target_column
                )
                selected_features = await self._select_features(features_with_updated_defaults,
                                                                data.feature_search, data.target_column,
                                                                enriched_feature_search_group_tables,
                                                                params, conn)
        finally:
            run_context.ddb_manager.detach(temp_db_name)

        # Get the original version of the selected features, and re-deduplicate their names
        # Some of the original deduplications may no longer be necessary.
        # Note that we need to use the selected feature and not the original feature at that index,
        # because it has updated defaults.
        def original_name(feature: Feature) -> str:
            index = next(idx for idx, gen in enumerate(deduplicated_candidate_features) if gen.feature.name == feature.name)
            return candidate_features[index].feature.name

        selected_features_with_original_names = [attrs.evolve(feature, name=original_name(feature)) for feature in selected_features]
        deduplicated_selected_features = deduplicate_feature_names(selected_features_with_original_names, existing_names=[data.target_column])

        with run_context.ddb_manager.cursor() as conn:
            enriched_eval_sink = DatasetSink.into_unqualified_duckdb_table('enriched_eval', conn) # TODO nonce name or parameter specifying storage target
            await params.enrich_runner.run_stream(deduplicated_selected_features, data.feature_eval, data.contexts,
                                                  enriched_eval_sink, params.evaluators, conn,
                                                  keep_input_columns=(data.target_column,),
                                                  deduplicate_names=False)
            features_with_eval_stats: list[FeatureWithFullStats[Feature, TK]] = await self._calculate_feature_stats_single_data(deduplicated_selected_features,
                                                                                           enriched_eval_sink.as_source(conn), data.target_column,
                                                                                           params, conn)

            enriched_test_sink = DatasetSink.into_unqualified_duckdb_table('enriched_test', conn)
            await params.enrich_runner.run_stream(deduplicated_selected_features, data.test, data.contexts,
                                                  enriched_test_sink, params.evaluators, conn,
                                                  keep_input_columns=(data.target_column,),
                                                  deduplicate_names=False)
            features_with_test_stats: list[FeatureWithFullStats[Feature, TK]] = await self._calculate_feature_stats_single_data(deduplicated_selected_features,
                                                                                           enriched_test_sink.as_source(conn), data.target_column,
                                                                                           params, conn)

            return FeatureSearchResults(tuple(features_with_eval_stats), tuple(features_with_test_stats))

    def _validate_input(self, data: FeatureSearchInputData, conn: DuckDBPyConnection) -> None:
        """Fail if the target column in any input dataset has missing values,
           or non-finite values if it is a float column.

        This requires reading the input datasets an extra time, which can be expensive
        if they are not stored in duckdb. In particular, without this, we could guarantee
        only reading the test dataset once (streaming it), and probably the full train dataset too.

        A future improvement can move the check to be done while streaming the dataset,
        but for now the decision was to check ahead of time.

        Because this can take unbounded time, it is run on the threadpool when this class calls it.
        """
        target_df = pl.DataFrame({'target': data.feature_search.data[data.target_column] })
        if target_df.filter(~pl.col('target').is_finite() | pl.col('target').is_null()).height > 0:
            raise ValueError('Target column may not contain missing values or non-finite float values (feature search dataset)')

        for name, source in [('feature evaluation', data.feature_eval), ('train', data.train), ('test', data.test)]:
            source_rel = source.to_duckdb(conn)
            expr = source_rel.filter(f'''"{data.target_column}" is null or "{data.target_column}" in ('nan'::float, 'inf'::float, '-inf'::float)''').aggregate('count(*)')
            count = expr.fetchone()
            match count:
                case (int(c),) if c > 0: raise ValueError(f'Target column may not contain missing values or non-finite float values ({name} dataset)')
                case _: pass


    def _deduplicate_generated_feature_names(self, features: Sequence[GeneratedFeature],
                                             existing_names: Sequence[str] = ()) -> list[GeneratedFeature]:
        return [GeneratedFeature(attrs.evolve(gen.feature, name=new_name), gen.has_good_defaults)
                if new_name != gen.feature.name else gen
                for gen, new_name in zip(features, deduplicate_strings([gen.feature.name for gen in features],
                                                                       existing=existing_names), strict=False)]

    async def _generate_features(self, conn: DuckDBPyConnection, data: FeatureSearchInputData,
                                 generators: Sequence[FeatureGenerator]) -> list[GeneratedFeature]:
        async with ScopedQueue[GeneratedFeature](maxsize=0) as queue: # maxsize=0 means unlimited
            sync_generators = [generator for generator in generators if isinstance(generator, SyncFeatureGenerator)]
            async_generators = [generator for generator in generators if not isinstance(generator, SyncFeatureGenerator)]

            if self.run_generators_concurrently:
                await asyncio.gather(
                    self._generate_sync(conn, queue, data, sync_generators),
                    self._generate_async(conn, queue, data, async_generators)
                )
            else:
                await self._generate_sync(conn, queue, data, sync_generators)
                await self._generate_async(conn, queue, data, async_generators)

            queue.close() # so that iteration will terminate when producing the list()
            return list(queue)

    async def _generate_sync(self, conn: DuckDBPyConnection, output_queue: Queue[GeneratedFeature], data: FeatureSearchInputData,
                             generators: list[SyncFeatureGenerator]) -> None:
        if not generators:
            return

        with conn.cursor() as cursor: # Cursor for new thread
            def sync_generate() -> None:
                for generator in generators:
                    _logger.info(f'Generating features with {generator=}')
                    for feature in generator.generate(data.feature_search, data.target_column, data.contexts, cursor):
                        output_queue.put(feature)
                    _logger.info(f'Done generating features with {generator=}')
            await asyncio.to_thread(sync_generate)

    async def _generate_async(self, conn: DuckDBPyConnection, output_queue: Queue[GeneratedFeature], data: FeatureSearchInputData,
                              generators: list[FeatureGenerator]) -> None:
        async def agenerate(generator: FeatureGenerator) -> None:
            _logger.info(f'Generating features with {generator=}')
            async for feature in generator.agenerate(data.feature_search, data.target_column, data.contexts, conn):
                await output_queue.aput(feature)
            _logger.info(f'Done generating features with {generator=}')

        if self.run_generators_concurrently:
            await asyncio.gather(*[agenerate(generator) for generator in generators])
        else:
            for generator in generators:
                await agenerate(generator)

    async def _enrich_in_batches_and_update_defaults(self, features: list[GeneratedFeature], dataset: Dataset,
                                                     contexts: TablesWithContextDefinitions, conn: DuckDBPyConnection,
                                                     params: FeatureSearchParams, target_table_base_name: DuckdbName,
                                                     target_column: str) -> tuple[list[DuckdbTable], list[Feature]]:
        """Enrich these features in batches of size up to self.max_features_enrich_batch_size,
        and return a table per batch.

        The first table also has the target column; the rest don't.
        """
        feature_groups = list(itertools.batched(features, self.max_features_enrich_batch_size))
        tables = []
        features_with_updated_defaults = []
        for index, feature_group in enumerate(feature_groups):
            keep_input_columns = (target_column,) if index == 0 else ()
            enriched_group = await params.enrich_runner.run([gen.feature for gen in feature_group],
                                                            dataset, contexts, params.evaluators,
                                                            conn, keep_input_columns=keep_input_columns,
                                                            deduplicate_names=False)
            group_table_name = attrs.evolve(target_table_base_name, name=f'{target_table_base_name.name}_{index}')
            DatasetSink.into_duckdb_table(group_table_name).write(
                DatasetSource.from_dataset(enriched_group), conn)
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

        _logger.info(f'Enriched {len(features)} features in {len(feature_groups)} batches')
        return (tables, features_with_updated_defaults)

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

    async def _select_features(self, candidate_features: list[Feature], feature_search: Dataset,
                               target_col: str, enriched_groups: list[DuckdbTable],
                               params: FeatureSearchParams[TK], conn: DuckDBPyConnection) -> list[Feature]:
        selector = params.selector
        if isinstance(selector, FeatureSelector):
            target_series = feature_search.data[target_col]
            features_with_data: list[tuple[Feature, DatasetSource]] = [
                (feature, DatasetSource.from_table(next(table for table in enriched_groups if feature.name in table.schema.names)))
                for feature in candidate_features
            ]
            features_with_stats: list[FeatureWithFullStats[Feature, TK]] = \
                await self._calculate_feature_stats(features_with_data, target_series, params, conn)

            if isinstance(selector, SyncFeatureSelector):
                def sync_select() -> list[Feature]:
                    for fws in features_with_stats:
                        selector.add_feature(fws)
                    return [fws.feature for fws in selector.select_final_features()]

                return await asyncio.to_thread(sync_select)

            else:
                for fws in features_with_stats:
                    await selector.aadd_feature(fws)
                return [fws.feature for fws in await selector.aselect_final_features()]

        else:
            # selector is EnrichedFeatureSelector. The first enriched table also contains the target column.
            enriched_source = self._join_tables(enriched_groups, conn)

            if isinstance(selector, SyncEnrichedFeatureSelector):
                with conn.cursor() as cursor: # for new thread
                    def select() -> list[Feature]:
                        return list(selector.select_features(candidate_features, enriched_source, target_col, cursor))
                    return await asyncio.to_thread(select)

            else:
                return list(await selector.aselect_features(candidate_features, enriched_source, target_col, conn))

    def _update_feature_defaults[F: Feature](self, feature: F, enriched: pl.Series) -> F:
        match feature:
            case IntFeature():
                return cast(F, attrs.evolve(feature, default_for_missing=int(cast(float, enriched.median()))))
            case BoolFeature():
                return cast(F, attrs.evolve(feature, default_for_missing=False))
            case CategoricalFeature():
                return cast(F, attrs.evolve(feature, default_for_missing=CategoricalFeature.other_category))
            case FloatFeature():
                finite_values = enriched.replace([math.inf, -math.inf], [None, None])
                max_val = cast(float, finite_values.max()) + 1
                min_val = cast(float, finite_values.min()) - 1
                substituted = enriched.replace([math.inf, -math.inf], [max_val, min_val])
                median = cast(float, substituted.median())
                return cast(F, attrs.evolve(feature, default_for_missing=median, default_for_nan=median,
                                            default_for_infinity=max_val, default_for_neg_infinity=min_val))
            case _:
                raise TypeError(f'Unexpected feature type {type(feature)}')


    async def _calculate_feature_stats(self, features_with_data: list[tuple[Feature, DatasetSource]],
                                       target_series: pl.Series,
                                       params: FeatureSearchParams[TK],
                                       conn: DuckDBPyConnection) -> list[FeatureWithFullStats[Feature, TK]]:
        # Stats calculators are always synchronous. We run them on a single thread, one feature at a time;
        # TODO use several threads
        with conn.cursor() as cursor: # for new thread
            def calculate() -> list[FeatureWithFullStats]:
                result = []
                for feature, data_source in features_with_data:
                    dataset = data_source.select(feature.name).to_dataset(cursor)
                    feature_stats = params.feature_stats_calculator.calculate_from_series(feature, dataset.data[feature.name])
                    relationship_stats = params.relationship_stats_calculator.calculate_from_series(feature, dataset.data[feature.name], target_series)
                    feature_with_stats = FeatureWithFullStats(feature, FullFeatureStats(feature_stats, relationship_stats))
                    result.append(feature_with_stats)
                return result

            return await asyncio.to_thread(calculate)

    async def _calculate_feature_stats_single_data(self, features: list[Feature],
                                                   dataset_source: DatasetSource, target_col: str,
                                                   params: FeatureSearchParams[TK],
                                                   conn: DuckDBPyConnection) -> list[FeatureWithFullStats[Feature, TK]]:
        target_source = dataset_source.select(target_col)
        with conn.cursor() as cursor:
            target_series = (await asyncio.to_thread(target_source.to_dataset, cursor)).data[target_col]

        return await self._calculate_feature_stats([(feature, dataset_source) for feature in features], target_series, params, conn)
