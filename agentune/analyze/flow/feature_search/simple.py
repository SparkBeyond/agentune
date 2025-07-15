import asyncio
import contextlib
import itertools
import logging
from contextlib import ExitStack
from typing import cast, override

import duckdb
from attrs import frozen

from agentune.analyze.context.base import TablesWithContextDefinitions, TableWithContextDefinitions
from agentune.analyze.core import setup
from agentune.analyze.core.database import DatabaseTable
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Schema
from agentune.analyze.feature.base import Feature, Regression, SyncFeature
from agentune.analyze.feature.describe.base import FeatureDescriber, SyncFeatureDescriber
from agentune.analyze.feature.eval.base import FeatureEvaluator, SyncFeatureEvaluator
from agentune.analyze.feature.gen.base import (
    FeatureGenerator,
    FeatureTransformer,
    SyncFeatureGenerator,
    SyncFeatureTransformer,
)
from agentune.analyze.feature.select.base import FeatureSelector, SyncFeatureSelector
from agentune.analyze.feature.stats.base import (
    FeatureStats,
    FeatureStatsCalculator,
    FeatureWithFullStats,
    FullFeatureStats,
    RelationshipStats,
    RelationshipStatsCalculator,
    SyncFeatureStatsCalculator,
    SyncRelationshipStatsCalculator,
)
from agentune.analyze.flow.duckdb import DuckdbManager
from agentune.analyze.flow.feature_search.base import (
    FeatureSearchDatasets,
    FeatureSearchFlow,
    RegressionFeatureSearchParams,
)
from agentune.analyze.util.queue import Queue

_logger = logging.getLogger(__name__)

@frozen
class EvaluatedFeatures:
    dataset: Dataset
    features: list[Feature]

#  _____     _____    _   _    _____   _______   _    _   _____  ______ 
# |  __ \   / ___ \  | \ | |  / ___ \ |__   __| | |  | | / ____||  ____|
# | |  | | | |   | | |  \| | | |   | |   | |    | |  | || (___  | |__   
# | |  | | | |   | | | . ` | | |   | |   | |    | |  | | \___ \ |  __|  
# | |__| | | |___| | | |\  | | |___| |   | |    | |__| | ____) || |____ 
# |_____/   \_____/  |_| \_|  \_____/    |_|     \____/ |_____/ |______|

class SimpleFeatureSearchFlow(FeatureSearchFlow):
    """Runs one unit at a time, holding all intermediate values in memory.
    Runs async units on a single asyncio event loop (which it creates), but only one at a time.
    This means all units run on the same (calling) thread and we don't have to manage moving values between threads.
    This is not usable in production; it's only for testing and development.
    """

    def __init__(self) -> None:
        self._queue_size = 2**31 # effectively infinite

    @override
    def run(self, params: RegressionFeatureSearchParams) -> tuple[FeatureWithFullStats[Feature, Regression], ...]:
        setup.setup()
        
        with ExitStack() as context_stack:
            ddb_manger = context_stack.enter_context(contextlib.closing(DuckdbManager.create('feature_search')))

            # Context generation
            with ddb_manger.cursor() as conn:
                contexts = self._generate_contexts(params.datasets, conn)

            # Feature generation
            generated_features = []
            for generator in params.feature_generators:
                _logger.info(f'Generating features with {generator}')
                generated_features.extend(self._generate_features(params.datasets, contexts, generator))
                _logger.info(f'Now have {len(generated_features)} features')
                        
            # Feature transformation
            if params.feature_transformer is None:
                transformed_features = generated_features
            else:
                transformer = params.feature_transformer # for mypy to realize this is a stable expression
                _logger.info(f'Transforming features with {transformer}')
                transformed_features = self._transform_features(params.datasets, contexts, generated_features, transformer)
                _logger.info(f'Created {len(transformed_features)} transformed features')
                transformed_features = generated_features + transformed_features
            del generated_features # free memory

            # Feature evaluation
            # This assumes a 'fallback' evaluator that can evaluate all features and comes last in the input list (TODO improve these APIs)
            features_by_evaluator = { evaluator: [
                feature for feature in transformed_features if evaluator.supports_feature(feature)
            ] for evaluator in params.feature_evaluators }
            del transformed_features # free memory

            evaluated_features = []
            for evaluator_cls, features in features_by_evaluator.items():
                with ddb_manger.cursor() as conn:
                    if issubclass(evaluator_cls, SyncFeatureEvaluator):
                        evaluated_features.extend(self._evaluate_sync_features(params.datasets, contexts, conn, evaluator_cls, cast(list[SyncFeature], features)))
                    else:
                        evaluated_features.extend(self._evaluate_async_features(params.datasets, contexts, conn, evaluator_cls, features))
                    _logger.info(f'Evaluated {len(features)} with {evaluator_cls.__name__}')

            del features_by_evaluator # free memory

            # Feature stats calculation
            features_with_stats = self._calculate_full_stats(params.datasets, params.feature_stats_calculator, params.relationship_stats_calculator, evaluated_features)
            _logger.info(f'Calculated stats for {len(features_with_stats)} features')
            del evaluated_features # free memory

            # Feature selection
            selected_features = self._select_features(params.feature_selector, features_with_stats)
            _logger.info(f'Selected {len(selected_features)} features')

            # Feature description
            if params.feature_describer is None:
                final_features = selected_features
            else:
                final_features = self._describe_features(params.feature_describer, selected_features)
                _logger.info('Recalculated feature descriptions')

            return tuple(final_features)
    
    def _generate_contexts(self, datasets: FeatureSearchDatasets, contexts_conn: duckdb.DuckDBPyConnection) -> TablesWithContextDefinitions:
        tables_with_contexts = []
        for context_source in datasets.context_sources:
            _logger.info(f'Ingesting context data from {context_source.name}')
            database_table = DatabaseTable(context_source.name, context_source.source.schema, 
                                           tuple(definition.index for definition in context_source.context_definitions))
            database_table.create(contexts_conn)
            context_source.source.open().to_duckdb(contexts_conn).insert_into(database_table.name)
            tables_with_contexts.append(TableWithContextDefinitions(database_table, tuple(context_source.context_definitions)))
        contexts = TablesWithContextDefinitions.from_list(tables_with_contexts)
        if len(contexts) > 0:
            _logger.info(f'All context data and objects: {contexts}')
        return contexts

    def _generate_features(self, datasets: FeatureSearchDatasets, contexts: TablesWithContextDefinitions, 
                           generator: FeatureGenerator) -> list[Feature]:
        if isinstance(generator, SyncFeatureGenerator):
            return list(generator.generate(datasets.feature_search, contexts))
        else:
            queue = Queue[Feature](self._queue_size)
            async def agenerate() -> None:
                await queue.aconsume(
                    generator.agenerate(datasets.feature_search, contexts)
                )
            asyncio.run(agenerate())
            queue.close()
            return list(queue)
        
    def _transform_features(self, datasets: FeatureSearchDatasets, contexts: TablesWithContextDefinitions, 
                            features: list[Feature], transformer: FeatureTransformer) -> list[Feature]:
        if isinstance(transformer, SyncFeatureTransformer):
            return [transformed for feature in features for transformed in transformer.transform(datasets.feature_search, contexts, feature)]
        else:
            queue = Queue[Feature](self._queue_size)
            async def atransform() -> None:
                for feature in features:
                    for transformed in await transformer.atransform(datasets.feature_search, contexts, feature):
                        await queue.aput(transformed)
            asyncio.run(atransform())
            queue.close()
            return list(queue)
        
    def _add_target_to_evaluated(self, input: Dataset, evaluated: Dataset, target_col: str) -> Dataset:
        target_series = input.data[target_col]
        target_field = input.schema[target_col]
        return Dataset(
            Schema((*evaluated.schema.cols, target_field)),
            evaluated.data.with_columns(target_series)
        )
        
        
    def _evaluate_sync_features(self, datasets: FeatureSearchDatasets, contexts: TablesWithContextDefinitions, 
                                conn: duckdb.DuckDBPyConnection,
                                evaluator_cls: type[SyncFeatureEvaluator], sync_features: list[SyncFeature]) -> list[EvaluatedFeatures]:
        feature_batch_size = 100
        evaluated_batches = []
        for sync_feature_batch in itertools.batched(sync_features, feature_batch_size):
            evaluator = evaluator_cls.for_features(list(sync_feature_batch))
            evaluated = evaluator.evaluate(datasets.feature_search, contexts, conn, include_originals=False)
            evaluated_with_target = self._add_target_to_evaluated(datasets.feature_search, evaluated, datasets.target_col)
            evaluated_batches.append(EvaluatedFeatures(evaluated_with_target, list(sync_feature_batch)))
        return evaluated_batches
    
    def _evaluate_async_features(self, datasets: FeatureSearchDatasets, contexts: TablesWithContextDefinitions, 
                                 conn: duckdb.DuckDBPyConnection,
                                 evaluator_cls: type[FeatureEvaluator], async_features: list[Feature]) -> list[EvaluatedFeatures]:
        feature_batch_size = 100
        queue = Queue[EvaluatedFeatures](self._queue_size)
        async def aevaluate() -> None:
            for async_feature_batch in itertools.batched(async_features, feature_batch_size):
                evaluator = evaluator_cls.for_features(list(async_feature_batch))
                evaluated = await evaluator.aevaluate(datasets.feature_search, contexts, conn, include_originals=False)
                evaluated_with_target = self._add_target_to_evaluated(datasets.feature_search, evaluated, datasets.target_col)  
                await queue.aput(EvaluatedFeatures(evaluated_with_target, list(async_feature_batch)))
        _logger.info('running async evaluate')
        asyncio.run(aevaluate())
        queue.close()
        return list(queue)
    
    def _calculate_full_stats(self, datasets: FeatureSearchDatasets, 
                              feature_stats_calculator: FeatureStatsCalculator[Feature],
                              relationship_stats_calculator: RelationshipStatsCalculator[Feature, Regression],
                              all_evaluated_features: list[EvaluatedFeatures]) -> list[FeatureWithFullStats[Feature, Regression]]:
        feature_stats_list = self._calculate_feature_stats(feature_stats_calculator, all_evaluated_features)
        relationship_stats_list = self._calculate_relationship_stats(datasets, relationship_stats_calculator, all_evaluated_features)
        return [FeatureWithFullStats(a[0], FullFeatureStats(a[1], b[1]))
                for a, b in zip(feature_stats_list, relationship_stats_list, strict=True)]
    
    def _calculate_feature_stats(self, 
                                 calculator: FeatureStatsCalculator, 
                                 all_evaluated_features: list[EvaluatedFeatures]) -> list[tuple[Feature, FeatureStats]]:
        if isinstance(calculator, SyncFeatureStatsCalculator):
            return [(feature, calculator.calculate_from_dataset(feature, evaluated_features.dataset, feature.name))
                    for evaluated_features in all_evaluated_features 
                    for feature in evaluated_features.features]
        else:
            queue = Queue[tuple[Feature, FeatureStats]](self._queue_size)
            async def acalculate() -> None:
                for evaluated_features in all_evaluated_features:
                    for feature in evaluated_features.features:
                        stats = await calculator.acalculate_from_dataset(feature, evaluated_features.dataset, feature.name)
                        await queue.aput((feature, stats))
            asyncio.run(acalculate())
            queue.close()
            return list(queue)
    
    def _calculate_relationship_stats(self, datasets: FeatureSearchDatasets, 
                                      calculator: RelationshipStatsCalculator[Feature, Regression], 
                                      all_evaluated_features: list[EvaluatedFeatures]) -> list[tuple[Feature, RelationshipStats[Feature, Regression]]]:
        if isinstance(calculator, SyncRelationshipStatsCalculator):
            return [(feature, calculator.calculate_from_dataset(feature, evaluated_features.dataset, feature.name, datasets.target_col))
                    for evaluated_features in all_evaluated_features 
                    for feature in evaluated_features.features]
        else:
            queue = Queue[tuple[Feature, RelationshipStats[Feature, Regression]]](self._queue_size)
            async def acalculate() -> None:
                for evaluated_features in all_evaluated_features:
                    for feature in evaluated_features.features:
                        stats = await calculator.acalculate_from_dataset(feature, evaluated_features.dataset, feature.name, datasets.target_col)
                        await queue.aput((feature, stats))
            asyncio.run(acalculate())
            queue.close()
            return list(queue)
        
        
    def _select_features(self, selector: FeatureSelector, features_with_stats: list[FeatureWithFullStats]) -> list[FeatureWithFullStats]:
        if isinstance(selector, SyncFeatureSelector):
            for feature_with_stats in features_with_stats:
                selector.add_feature(feature_with_stats)
            return list(selector.select_final_features())
        else:
            async def aselect() -> list[FeatureWithFullStats]:
                for feature_with_stats in features_with_stats:
                    await selector.aadd_feature(feature_with_stats)
                return list(await selector.aselect_final_features())
            return asyncio.run(aselect())
        
    def _describe_features(self, describer: FeatureDescriber, selected_features: list[FeatureWithFullStats]) -> list[FeatureWithFullStats]:
        if isinstance(describer, SyncFeatureDescriber):
            return [FeatureWithFullStats(describer.describe(feature_with_stats.feature), feature_with_stats.stats) 
                    for feature_with_stats in selected_features]
        else:
            async def adescribe() -> list[FeatureWithFullStats]:
                return [FeatureWithFullStats(await describer.adescribe(feature_with_stats.feature), feature_with_stats.stats) 
                        for feature_with_stats in selected_features]
            return asyncio.run(adescribe())
        
    def _copy_queue[T](self, queue: Queue[T]) -> Queue[T]:
        # A Janus queue's async side can't interact with two different event loops over its lifetime,
        # so as an ugly hack, we copy to a new queue that can interact with the next async event loop
        new_queue = Queue[T](queue.maxsize)
        new_queue.consume(iter(queue))
        return new_queue

