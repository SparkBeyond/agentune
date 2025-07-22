import asyncio
import itertools
import logging
from typing import cast, override

import duckdb
from attrs import frozen

from agentune.analyze.context.base import TablesWithContextDefinitions
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Schema
from agentune.analyze.feature.base import Feature, SyncFeature, TargetKind
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
from agentune.analyze.run.base import RunContext
from agentune.analyze.run.feature_search.base import (
    FeatureSearchInputData,
    FeatureSearchParams,
    FeatureSearchResults,
    FeatureSearchRunner,
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

class SimpleFeatureSearchRunner(FeatureSearchRunner):
    """Runs one unit at a time, holding all intermediate values in memory.
    Runs async units on a single asyncio event loop (which it creates), but only one at a time.
    This means all units run on the same (calling) thread and we don't have to manage moving values between threads.
    This is not usable in production; it's only for testing and development.
    """

    def __init__(self) -> None:
        self._queue_size = 2**31 # effectively infinite

    @override
    def run[TK: TargetKind](self, context: RunContext, input_data: FeatureSearchInputData, 
                            params: FeatureSearchParams[TK]) -> FeatureSearchResults[TK]: 

        # Feature generation
        generated_features = []
        for generator in params.generators:
            _logger.info(f'Generating features with {generator}')
            generated_features.extend(self._generate_features(input_data, input_data.contexts, generator))
            _logger.info(f'Now have {len(generated_features)} features')
                    
        # Feature evaluation
        # This assumes a 'fallback' evaluator that can evaluate all features and comes last in the input list (TODO improve these APIs)
        features_by_evaluator = { evaluator: [
            feature for feature in generated_features if evaluator.supports_feature(feature)
        ] for evaluator in params.evaluators }
        del generated_features # free memory

        evaluated_features = []
        for evaluator_cls, features in features_by_evaluator.items():
            with context.ddb_manager.cursor() as conn:
                if issubclass(evaluator_cls, SyncFeatureEvaluator):
                    evaluated_features.extend(self._evaluate_sync_features(input_data, input_data.contexts, conn, evaluator_cls, cast(list[SyncFeature], features)))
                else:
                    evaluated_features.extend(self._evaluate_async_features(input_data, input_data.contexts, conn, evaluator_cls, features))
                _logger.info(f'Evaluated {len(features)} with {evaluator_cls.__name__} on {input_data.feature_search.data.height} rows')

        del features_by_evaluator # free memory
        # _logger.info(f'Example feature stats: {evaluated_features[0].dataset}')

        # Feature stats calculation
        features_with_stats = self._calculate_full_stats(input_data, params.feature_stats_calculator, params.relationship_stats_calculator, evaluated_features)
        _logger.info(f'Calculated stats for {len(features_with_stats)} features')
        del evaluated_features # free memory

        # Feature selection
        selected_features = self._select_features(params.selector, features_with_stats)
        _logger.info(f'Selected {len(selected_features)} features')

        # TODO evaluate on the right datasets; right now we only have the features as evaluated on the feature search dataset
        results = FeatureSearchResults(
            features_with_train_stats=tuple(selected_features),
            features_with_test_stats=tuple(selected_features),
        )
        
        return results
    
    def _generate_features(self, datasets: FeatureSearchInputData, contexts: TablesWithContextDefinitions, 
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
        
    def _transform_features(self, datasets: FeatureSearchInputData, contexts: TablesWithContextDefinitions, 
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
        
        
    def _evaluate_sync_features(self, datasets: FeatureSearchInputData, contexts: TablesWithContextDefinitions, 
                                conn: duckdb.DuckDBPyConnection,
                                evaluator_cls: type[SyncFeatureEvaluator], sync_features: list[SyncFeature]) -> list[EvaluatedFeatures]:
        feature_batch_size = 100
        evaluated_batches = []
        for sync_feature_batch in itertools.batched(sync_features, feature_batch_size):
            evaluator = evaluator_cls.for_features(list(sync_feature_batch))
            evaluated = evaluator.evaluate(datasets.feature_search, contexts, conn, include_originals=False)
            evaluated_with_target = self._add_target_to_evaluated(datasets.feature_search, evaluated, datasets.target_column)
            evaluated_batches.append(EvaluatedFeatures(evaluated_with_target, list(sync_feature_batch)))
        return evaluated_batches
    
    def _evaluate_async_features(self, datasets: FeatureSearchInputData, contexts: TablesWithContextDefinitions, 
                                 conn: duckdb.DuckDBPyConnection,
                                 evaluator_cls: type[FeatureEvaluator], async_features: list[Feature]) -> list[EvaluatedFeatures]:
        feature_batch_size = 100
        queue = Queue[EvaluatedFeatures](self._queue_size)
        async def aevaluate() -> None:
            for async_feature_batch in itertools.batched(async_features, feature_batch_size):
                evaluator = evaluator_cls.for_features(list(async_feature_batch))
                evaluated = await evaluator.aevaluate(datasets.feature_search, contexts, conn, include_originals=False)
                evaluated_with_target = self._add_target_to_evaluated(datasets.feature_search, evaluated, datasets.target_column)  
                await queue.aput(EvaluatedFeatures(evaluated_with_target, list(async_feature_batch)))
        _logger.info('running async evaluate')
        asyncio.run(aevaluate())
        queue.close()
        return list(queue)
    
    def _calculate_full_stats[TK: TargetKind](self, datasets: FeatureSearchInputData, 
                              feature_stats_calculator: FeatureStatsCalculator[Feature],
                              relationship_stats_calculator: RelationshipStatsCalculator[Feature, TK],
                              all_evaluated_features: list[EvaluatedFeatures]) -> list[FeatureWithFullStats[Feature, TK]]:
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
    
    def _calculate_relationship_stats[TK: TargetKind](self, datasets: FeatureSearchInputData, 
                                      calculator: RelationshipStatsCalculator[Feature, TK], 
                                      all_evaluated_features: list[EvaluatedFeatures]) -> list[tuple[Feature, RelationshipStats[Feature, TK]]]:
        if isinstance(calculator, SyncRelationshipStatsCalculator):
            return [(feature, calculator.calculate_from_dataset(feature, evaluated_features.dataset, feature.name, datasets.target_column))
                    for evaluated_features in all_evaluated_features 
                    for feature in evaluated_features.features]
        else:
            queue = Queue[tuple[Feature, RelationshipStats[Feature, TK]]](self._queue_size)
            async def acalculate() -> None:
                for evaluated_features in all_evaluated_features:
                    for feature in evaluated_features.features:
                        stats = await calculator.acalculate_from_dataset(feature, evaluated_features.dataset, feature.name, datasets.target_column)
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

