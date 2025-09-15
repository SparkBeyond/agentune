from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self, final

from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.core.database import DuckdbName, DuckdbTable
from agentune.analyze.core.dataset import Dataset, DatasetSource
from agentune.analyze.core.threading import CopyToThread
from agentune.analyze.feature.base import Feature
from agentune.analyze.feature.eval.base import FeatureEvaluator
from agentune.analyze.feature.eval.universal import (
    UniversalAsyncFeatureEvaluator,
    UniversalSyncFeatureEvaluator,
)
from agentune.analyze.feature.gen.base import FeatureGenerator
from agentune.analyze.feature.problem import Problem, ProblemDescription
from agentune.analyze.feature.select.base import EnrichedFeatureSelector, FeatureSelector
from agentune.analyze.feature.stats.base import (
    FeatureWithFullStats,
)
from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.analyze.run.base import RunContext
from agentune.analyze.run.enrich.base import EnrichRunner
from agentune.analyze.run.enrich.impl import EnrichRunnerImpl
from agentune.analyze.run.ingest.sampling import SplitDuckdbTable


@final
@frozen
class FeatureSearchInputData(CopyToThread):
    feature_search: Dataset # Small dataset for feature generators, held in memory
    feature_eval: DatasetSource 
    train: DatasetSource # Includes the feature_search and feature_eval datasets
    test: DatasetSource
    join_strategies: TablesWithJoinStrategies

    def __attrs_post_init__(self) -> None:
        if self.train.schema != self.test.schema:
            raise ValueError('Train schema must match test schema')
        if self.train.schema != self.feature_search.schema:
            raise ValueError('Train schema must match feature search schema')
        if self.train.schema != self.feature_eval.schema:
            raise ValueError('Train schema must match feature eval schema')

    @staticmethod
    def from_split_table(split_table: SplitDuckdbTable,
                         join_strategies: TablesWithJoinStrategies,
                         conn: DuckDBPyConnection) -> FeatureSearchInputData:
        return FeatureSearchInputData(
            feature_search=split_table.feature_search().to_dataset(conn),
            train=split_table.train(), 
            test=split_table.test(),
            feature_eval=split_table.feature_eval(),
            join_strategies=join_strategies
        )

    def copy_to_thread(self) -> Self:
        return FeatureSearchInputData(
            self.feature_search.copy_to_thread(),
            self.feature_eval.copy_to_thread(),
            self.train.copy_to_thread(),
            self.test.copy_to_thread(),
            self.join_strategies
        )


@frozen
class FeatureSearchParams:
    """Non-data arguments to feature search.

    Args:
        store_enriched_train: if not None, the final features computed on the train dataset will be stored in the named table.
                              and remain available after the feature search completes. If this table already exists,
                              it will be replaced.
                              This is the data that FeatureSearchResults.features_with_train_stats is computed on.
                              If None, the data will be stored in a temporary table and deleted before feature search
                              completes.
        store_enriched_test:  as above, for the test dataset.
    """
    problem_description: ProblemDescription
    generators: tuple[FeatureGenerator, ...]
    selector: FeatureSelector | EnrichedFeatureSelector
    # Must always include at least one evaluator willing to handle every feature generated.
    # Normally this means including the two universal evaluators at the end of the list.
    # Evaluators are tried in the order in which they appear.
    evaluators: tuple[type[FeatureEvaluator], ...] = (UniversalSyncFeatureEvaluator, UniversalAsyncFeatureEvaluator)
    enrich_runner: EnrichRunner = EnrichRunnerImpl()
    store_enriched_train: DuckdbName | None = None
    store_enriched_test: DuckdbName | None = None
    max_classes: int = 20

@frozen
class FeatureSearchResults:
    """Args:
    enriched_train: if FeatureSearchParams.store_enriched_train was given, this is the table where the data was stored.
                    This is the data that features_with_train_stats was computed on.
                    This table includes the target column and the enriched feature columns, but not the other
                    columns of the original input.
    enriched_test:  as above, for the test dataset.
    """
    problem: Problem
    features_with_train_stats: tuple[FeatureWithFullStats, ...]
    features_with_test_stats: tuple[FeatureWithFullStats, ...]
    enriched_train: DuckdbTable | None = None
    enriched_test: DuckdbTable | None = None

    def __attrs_post_init__(self) -> None:
        if tuple(f.feature for f in self.features_with_train_stats) != tuple(f.feature for f in self.features_with_test_stats):
            raise ValueError('Features with train stats must match features with test stats')

    @property
    def features(self) -> tuple[Feature, ...]:
        return tuple(f.feature for f in self.features_with_test_stats)


class FeatureSearchRunner(ABC):
    """The current implementation is not specialized per TargetKind, but including the type parameter in the class signature
    makes the code much simpler than passing it to every method along the way.
    """

    @abstractmethod
    async def run(self, run_context: RunContext, data: FeatureSearchInputData,
                  params: FeatureSearchParams) -> FeatureSearchResults:
        """The feature search algorithm composes the components in `params`:

        1. Generate candidate features using `params.generators` on `data.feature_search`
        2. Impute default values, using the feature search dataset, if the generator didn't provide default values
        3. Select the final features using `params.selector` using the enriched feature search dataset
        4. Enrich `data.feature_evaluation` and `data.test` and calculate statistics on those two datasets

        Raises:
            NoFeaturesFoundError: if no generator returned any candidate features
        """
        ...

class NoFeaturesFoundError(Exception): pass
