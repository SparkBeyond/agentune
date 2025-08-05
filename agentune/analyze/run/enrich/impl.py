import asyncio
import logging
from collections.abc import Sequence
from typing import cast, override

import polars as pl
from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.context.base import TablesWithContextDefinitions
from agentune.analyze.core.dataset import (
    Dataset,
    DatasetSink,
    DatasetSource,
    DatasetSourceFromIterable,
)
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.feature.base import Feature, SyncFeature
from agentune.analyze.feature.dedup_names import deduplicate_feature_names
from agentune.analyze.feature.eval.base import FeatureEvaluator, SyncFeatureEvaluator
from agentune.analyze.run.enrich.base import EnrichRunner
from agentune.analyze.util.queue import ScopedQueue

_logger = logging.getLogger(__name__)


@frozen
class _EvaluatedFeatures:
    """Features evaluated together with their result dataset."""
    dataset: Dataset
    features: list[Feature]

class EnrichRunnerImpl(EnrichRunner):
    """Simple implementation of EnrichRunner that evaluates features by evaluator groups.
    
    Groups features by evaluator, runs async evaluators concurrently and sync evaluators
    sequentially on separate threads. Returns dataset with columns in original feature order.
    """

    @override
    async def run(self, features: Sequence[Feature], dataset: Dataset, contexts: TablesWithContextDefinitions,
                  evaluators: Sequence[type[FeatureEvaluator]], conn: DuckDBPyConnection,
                  deduplicate_names: bool = True) -> Dataset:
        features = self._maybe_deduplicate_feature_names(features, deduplicate_names)
        features_by_evaluator = self._group_features_by_evaluator(features, evaluators)
        all_evaluated = await self._evaluate_all_groups(
            conn, dataset, contexts, features_by_evaluator
        )
        return self._combine_evaluated_features(all_evaluated, features)

    @override
    async def run_stream(self, features: Sequence[Feature], dataset_source: DatasetSource, contexts: TablesWithContextDefinitions,
                         dataset_sink: DatasetSink, evaluators: Sequence[type[FeatureEvaluator]], conn: DuckDBPyConnection,
                         deduplicate_names: bool = True) -> None:
        """This 'Simple' (i.e. naive) runner has performance issues:
        - The buffer size (i.e. the size of the Datasets in the DatasetSource stream) is set by the `input` creator,
          possibly the user, and can't be adjusted to the features or evaluators involved
        - Async features evaluated in parallel means we wait for the slowest to finish before moving to the
          next batch. If the slowest isn't consistent (but is due to e.g. occasional timeouts that we'll retry),
          this can be much less efficient than letting the other features start evaluating the next batch
          while waiting for the slow ones. A sliding window over rows being processed would be better.
        """
        renamed_features = self._maybe_deduplicate_feature_names(features, deduplicate_names)
        schema = Schema(tuple(Field(feature.name, feature.dtype) for feature in renamed_features))

        # Separate input and output connections to run on different threads
        with (conn.cursor() as input_conn, conn.cursor() as output_conn, \
                ScopedQueue[Dataset]() as input_queue, ScopedQueue[Dataset]() as output_queue):

            def consume_input() -> None:
                input_queue.consume(dataset_source.copy_to_thread().open(input_conn))
            input_task = asyncio.create_task(asyncio.to_thread(consume_input))

            # This is an illegal DatasetSource which can only be consumed once, but that's all a DatasetSink needs
            output_source = DatasetSourceFromIterable(schema, output_queue)
            output_task = asyncio.create_task(asyncio.to_thread(dataset_sink.write, output_source, output_conn))

            async for dataset in input_queue:
                result = await self.run(features, dataset, contexts, evaluators, conn, deduplicate_names)
                await output_queue.aput(result)
            output_queue.close()
            await input_queue.await_empty()

            await input_task
            await output_task

            await input_queue.await_empty()
            await output_queue.await_empty()

    def _maybe_deduplicate_feature_names(self, features: Sequence[Feature], deduplicate_names: bool) -> Sequence[Feature]:
        if deduplicate_names:
            return deduplicate_feature_names(features)
        else:
            names = [f.name for f in features]
            if len(set(names)) != len(names):
                raise ValueError('Duplicate feature names found and deduplicate_names=False')
            return features

    def _group_features_by_evaluator(
        self,
        features: Sequence[Feature],
        evaluators: Sequence[type[FeatureEvaluator]]
    ) -> dict[type[FeatureEvaluator], list[Feature]]:
        """Group features by the first evaluator that can handle them."""
        features_by_evaluator = {}
        remaining_features = list(features)

        for evaluator_cls in evaluators:
            supported_features = [f for f in remaining_features if evaluator_cls.supports_feature(f)]
            if supported_features:
                features_by_evaluator[evaluator_cls] = supported_features
                remaining_features = [f for f in remaining_features if f not in supported_features]

        if remaining_features:
            raise ValueError(f'No evaluator found for features: {[f.name for f in remaining_features]}')

        return features_by_evaluator

    async def _evaluate_all_groups(
        self,
        conn: DuckDBPyConnection,
        dataset: Dataset,
        contexts: TablesWithContextDefinitions,
        features_by_evaluator: dict[type[FeatureEvaluator], list[Feature]]
    ) -> Sequence[_EvaluatedFeatures]:
        """Evaluate all feature groups, running all async evaluators concurrently
        and all sync evaluators serially (in parallel with the async ones).
        """
        async_groups = []
        sync_groups = []

        for evaluator_cls, features in features_by_evaluator.items():
            if issubclass(evaluator_cls, SyncFeatureEvaluator):
                sync_groups.append((evaluator_cls, cast(list[SyncFeature], features)))
            else:
                async_groups.append((evaluator_cls, features))

        async_tasks = [asyncio.create_task(self._evaluate_async_features(conn, dataset, contexts, evaluator_cls, features))
                       for evaluator_cls, features in async_groups]

        if sync_groups:
            with conn.cursor() as sync_cursor:
                sync_evaluated = await asyncio.to_thread(
                    self._evaluate_all_sync_features, dataset.copy_to_thread(), contexts, sync_cursor, sync_groups
                )
        else:
            sync_evaluated = []

        async_evaluated = await asyncio.gather(*async_tasks)
        return sync_evaluated + async_evaluated

    async def _evaluate_async_features(
        self,
        conn: DuckDBPyConnection,
        dataset: Dataset,
        contexts: TablesWithContextDefinitions,
        evaluator_cls: type[FeatureEvaluator],
        features: list[Feature]
    ) -> _EvaluatedFeatures:
        """Evaluate async features as a single group."""
        evaluator = evaluator_cls.for_features(features)
        evaluated = await evaluator.aevaluate(dataset, contexts, conn)
        _logger.info(f'Evaluated {len(features)} async features with {evaluator_cls.__name__}')
        return _EvaluatedFeatures(evaluated, features)

    def _evaluate_all_sync_features(
        self,
        dataset: Dataset,
        contexts: TablesWithContextDefinitions,
        conn: DuckDBPyConnection,
        sync_groups: list[tuple[type[SyncFeatureEvaluator], list[SyncFeature]]]
    ) -> list[_EvaluatedFeatures]:
        """Evaluate all sync features sequentially (called from async thread)."""
        all_evaluated = []

        for evaluator_cls, features in sync_groups:
            evaluator = evaluator_cls.for_features(features)
            evaluated = evaluator.evaluate(dataset, contexts, conn)
            all_evaluated.append(_EvaluatedFeatures(evaluated, cast(list[Feature], features)))
            _logger.info(f'Evaluated {len(features)} sync features with {evaluator_cls.__name__}')

        return all_evaluated

    def _combine_evaluated_features(
        self,
        all_evaluated: Sequence[_EvaluatedFeatures],
        original_features: Sequence[Feature]
    ) -> Dataset:
        """Combine evaluated features into single dataset, preserving original feature order."""
        feature_to_column = {}
        feature_to_field = {}

        for evaluated in all_evaluated:
            for feature in evaluated.features:
                column = evaluated.dataset.data.get_column(feature.name)
                field = evaluated.dataset.schema[feature.name]
                feature_to_column[feature] = column
                feature_to_field[feature] = field

        result_columns = []
        result_fields = []

        for feature in original_features:
            if feature not in feature_to_column:
                raise ValueError(f'Feature {feature.name} was not evaluated')

            result_columns.append(feature_to_column[feature])
            result_fields.append(feature_to_field[feature])

        result_data = pl.DataFrame(result_columns)
        result_schema = Schema(tuple(result_fields))

        return Dataset(result_schema, result_data)
