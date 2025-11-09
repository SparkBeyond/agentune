import asyncio
import logging
from collections.abc import Sequence
from typing import cast, override

import polars as pl
from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.core.dataset import (
    Dataset,
    DatasetSink,
    DatasetSource,
    DatasetSourceFromIterable,
)
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.feature.base import Feature, SyncFeature
from agentune.analyze.feature.compute.base import FeatureComputer, SyncFeatureComputer
from agentune.analyze.feature.compute.limits import async_features_compute_limit_context
from agentune.analyze.feature.dedup_names import deduplicate_feature_names
from agentune.analyze.progress.base import ProgressStage, stage_scope
from agentune.analyze.run.enrich.base import EnrichRunner
from agentune.analyze.util.queue import ScopedQueue

_logger = logging.getLogger(__name__)


@frozen
class _ComputedFeatures:
    """Features computed together with their result dataset."""
    dataset: Dataset
    features: list[Feature]


@frozen
class EnrichRunnerImpl(EnrichRunner):
    """Simple implementation of EnrichRunner that computes features by feature computer groups.
    
    Groups features by feature computer, runs async computers concurrently and sync computers
    sequentially on separate threads. Returns dataset with columns in original feature order.

    Args:
        max_async_features_compute: the max number of async features being computed at a time.

                                 We assume async features are efficient (in the sense that they don't block threads needlessly
                                 and they don't spawn synchronous in-process work); this limit is meant to avoid creating
                                 too many asyncio tasks at once, which uses memory unnecessarily. It should be kept
                                 high enough to saturate whatever external async resource these features are accessing.

                                 In the future, this limit may become more specific to particular feature types.
    """

    max_async_features_compute: int = 1000

    @override
    async def run(self, features: Sequence[Feature], dataset: Dataset,
                  feature_computers: Sequence[type[FeatureComputer]], conn: DuckDBPyConnection,
                  keep_input_columns: Sequence[str] = (),
                  deduplicate_names: bool = True) -> Dataset:
        with stage_scope('Compute cells (features*rows)', 0) as cells_progress:
            return await self._run(features, dataset, feature_computers, conn, cells_progress, keep_input_columns, deduplicate_names)

    async def _run(self, features: Sequence[Feature], dataset: Dataset,
                   feature_computers: Sequence[type[FeatureComputer]], conn: DuckDBPyConnection, cells_progress: ProgressStage,
                   keep_input_columns: Sequence[str] = (),
                   deduplicate_names: bool = True) -> Dataset:
        with async_features_compute_limit_context(self.max_async_features_compute):
            features = self._maybe_deduplicate_feature_names(features, deduplicate_names)
            features_by_computer = self._group_features_by_computer(features, feature_computers)
            all_computed = await self._compute_all_groups(
                conn, dataset, features_by_computer, cells_progress
            )
            return self._combine_computed_features(all_computed, features,
                                                   dataset.select(*keep_input_columns) if keep_input_columns else None)

    @override
    async def run_stream(self, features: Sequence[Feature], dataset_source: DatasetSource,
                         dataset_sink: DatasetSink, feature_computers: Sequence[type[FeatureComputer]], conn: DuckDBPyConnection,
                         keep_input_columns: Sequence[str] = (),
                         deduplicate_names: bool = True) -> None:
        """This naive implementation has performance issues:
        - The buffer size (i.e. the size of the Datasets in the DatasetSource stream) is set by the `dataset_source` creator,
          possibly the user, and can't be adjusted to the features or feature computers involved
        - Async features computed in parallel means we wait for the slowest to finish before moving to the
          next batch. If the slowest isn't consistent (but is due to e.g. occasional timeouts that we'll retry),
          this can be much less efficient than letting the other features start computing the next batch
          while waiting for the slow ones. A sliding window over rows being processed would be better.
        """
        renamed_features = self._maybe_deduplicate_feature_names(features, deduplicate_names)
        schema = Schema(tuple(field for field in dataset_source.schema.cols if field.name in keep_input_columns) + \
            tuple(Field(feature.name, feature.dtype) for feature in renamed_features))

        total_rows = dataset_source.cheap_size(conn)
        total_cells = total_rows * len(features) if total_rows is not None else None
        with stage_scope(f'Enrich rows with {len(features)} features', 0, total_rows) as rows_progress, \
             stage_scope('Compute cells (features*rows)', 0, total_cells) as cells_progress:
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
                    result = await self._run(features, dataset, feature_computers, conn, cells_progress, keep_input_columns, deduplicate_names)
                    rows_progress.increment_count(result.height)
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

    def _group_features_by_computer(
        self,
        features: Sequence[Feature],
        feature_computers: Sequence[type[FeatureComputer]]
    ) -> dict[type[FeatureComputer], list[Feature]]:
        """Group features by the first feature computer that can handle them."""
        features_by_computer = {}
        remaining_features = list(features)

        for feature_computer_cls in feature_computers:
            supported_features = [f for f in remaining_features if feature_computer_cls.supports_feature(f)]
            if supported_features:
                features_by_computer[feature_computer_cls] = supported_features
                remaining_features = [f for f in remaining_features if f not in supported_features]

        if remaining_features:
            raise ValueError(f'No feature computer found for features: {[f.name for f in remaining_features]}')

        return features_by_computer

    async def _compute_all_groups(
        self,
        conn: DuckDBPyConnection,
        dataset: Dataset,
        features_by_computer: dict[type[FeatureComputer], list[Feature]],
        cells_progress: ProgressStage
    ) -> Sequence[_ComputedFeatures]:
        """Compute all feature groups, running all async feature computers concurrently
        and all sync computers serially (in parallel with the async ones).
        """
        async_groups = []
        sync_groups = []

        for feature_computer_cls, features in features_by_computer.items():
            if issubclass(feature_computer_cls, SyncFeatureComputer):
                sync_groups.append((feature_computer_cls, cast(list[SyncFeature], features)))
            else:
                async_groups.append((feature_computer_cls, features))

        async_tasks = [asyncio.create_task(self._compute_async_features(conn, dataset, feature_computer_cls, features, cells_progress))
                       for feature_computer_cls, features in async_groups]

        if sync_groups:
            with conn.cursor() as sync_cursor:
                sync_computed = await asyncio.to_thread(
                    self._compute_all_sync_features, dataset.copy_to_thread(), sync_cursor, sync_groups, cells_progress
                )
        else:
            sync_computed = []

        async_computed = await asyncio.gather(*async_tasks)
        return sync_computed + async_computed

    async def _compute_async_features(
        self,
        conn: DuckDBPyConnection,
        dataset: Dataset,
        feature_computer_cls: type[FeatureComputer],
        features: list[Feature],
        cells_progress: ProgressStage
    ) -> _ComputedFeatures:
        """Compute async features as a single group."""
        feature_computer = feature_computer_cls.for_features(features)
        computed = await feature_computer.acompute(dataset, conn, cells_progress)
        return _ComputedFeatures(computed, features)

    def _compute_all_sync_features(
        self,
        dataset: Dataset,
        conn: DuckDBPyConnection,
        sync_groups: list[tuple[type[SyncFeatureComputer], list[SyncFeature]]],
        cells_progress: ProgressStage
    ) -> list[_ComputedFeatures]:
        """Compute all sync features sequentially (called from async thread)."""
        all_computed = []

        for computer_cls, features in sync_groups:
            computer = computer_cls.for_features(features)
            computed = computer.compute(dataset, conn, cells_progress)
            all_computed.append(_ComputedFeatures(computed, cast(list[Feature], features)))

        return all_computed

    def _combine_computed_features(
        self,
        all_computed: Sequence[_ComputedFeatures],
        original_features: Sequence[Feature],
        keep_input_dataset: Dataset | None
    ) -> Dataset:
        """Combine computed features into single dataset, preserving original feature order."""
        feature_to_column = {}
        feature_to_field = {}

        for computed in all_computed:
            for feature in computed.features:
                column = computed.dataset.data.get_column(feature.name)
                field = computed.dataset.schema[feature.name]
                feature_to_column[feature] = column
                feature_to_field[feature] = field

        result_columns = []
        result_fields = []

        if keep_input_dataset:
            for field in keep_input_dataset.schema.cols:
                result_fields.append(field)
                result_columns.append(keep_input_dataset.data[field.name])

        for feature in original_features:
            if feature not in feature_to_column:
                raise ValueError(f'Feature {feature.name} was not computed')

            result_columns.append(feature_to_column[feature])
            result_fields.append(feature_to_field[feature])

        result_data = pl.DataFrame(result_columns)
        result_schema = Schema(tuple(result_fields))

        return Dataset(result_schema, result_data)
