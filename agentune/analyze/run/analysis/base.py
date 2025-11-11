from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self, final

from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.core.database import DuckdbManager, DuckdbName, DuckdbTable
from agentune.analyze.core.dataset import Dataset, DatasetSource
from agentune.analyze.core.threading import CopyToThread
from agentune.analyze.feature.base import Feature
from agentune.analyze.feature.compute.base import FeatureComputer
from agentune.analyze.feature.compute.universal import (
    UniversalAsyncFeatureComputer,
    UniversalSyncFeatureComputer,
)
from agentune.analyze.feature.gen.base import FeatureGenerator
from agentune.analyze.feature.problem import Problem, ProblemDescription
from agentune.analyze.feature.select.base import EnrichedFeatureSelector, FeatureSelector
from agentune.analyze.feature.stats.base import (
    FeatureWithFullStats,
)
from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.analyze.run.enrich.base import EnrichRunner
from agentune.analyze.run.enrich.impl import EnrichRunnerImpl
from agentune.analyze.run.ingest.sampling import SplitDuckdbTable


@final
@frozen
class AnalyzeInputData(CopyToThread):
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
                         conn: DuckDBPyConnection) -> AnalyzeInputData:
        return AnalyzeInputData(
            feature_search=split_table.feature_search().to_dataset(conn),
            train=split_table.train(), 
            test=split_table.test(),
            feature_eval=split_table.feature_eval(),
            join_strategies=join_strategies
        )

    def copy_to_thread(self) -> Self:
        return AnalyzeInputData(
            self.feature_search.copy_to_thread(),
            self.feature_eval.copy_to_thread(),
            self.train.copy_to_thread(),
            self.test.copy_to_thread(),
            self.join_strategies
        )

@frozen
class UniqueTableName:
    """A request to create a unique (nonexistent) table name with the given prefix."""
    basename: str | DuckdbName

@frozen
class AnalyzeParams:
    """Non-data arguments to the analyzer.

    Args:
        store_enriched_train: if not None, the final features computed on the train dataset will be stored in the named table
                              and remain available after the analysis completes. If this table already exists,
                              it will be replaced.
                              If a UniqueTableName is passed, a random suffix will be appended to its `basename`
                              to make sure the table does not replace an existing one.
                              If None, the data will be stored in a temporary table and deleted before analysis completes.

                              This is the data that AnalyzeResults.features_with_train_stats is computed on.
        store_enriched_test:  as above, for the test dataset.
        max_features_to_select: maximum number of features to return from the analysis.
        max_classes:          Maximum number of distinct target values allowed for classification problems;
                              if more values are present in the train dataset, analysis will fail.
    """
    store_enriched_train: str | DuckdbName | UniqueTableName | None = UniqueTableName('enriched_train')
    store_enriched_test: str | DuckdbName | UniqueTableName | None = UniqueTableName('enriched_test')
    max_classes: int = 20
    max_features_to_select: int = 60

@frozen
class AnalyzeComponents:
    generators: tuple[FeatureGenerator, ...]
    selector: FeatureSelector | EnrichedFeatureSelector
    # Must always include at least one feature computer willing to handle every feature generated.
    # Normally this means including the two universal computers at the end of the list.
    # Feature computeres are tried in the order in which they appear.
    feature_computers: tuple[type[FeatureComputer], ...] = (UniversalSyncFeatureComputer, UniversalAsyncFeatureComputer)
    enrich_runner: EnrichRunner = EnrichRunnerImpl()


@frozen
class AnalyzeResults:
    """Args:
    enriched_train: if `AnalyzeParams.store_enriched_train` was given, this is the table where the data was stored.
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


class AnalyzeRunner(ABC):
    @abstractmethod
    async def run(self, ddb_manager: DuckdbManager, data: AnalyzeInputData,
                  params: AnalyzeParams, components: AnalyzeComponents,
                  problem_description: ProblemDescription) -> AnalyzeResults:
        """The analysis algorithm composes the components in `params`:

        1. Generate candidate features using `params.generators` on `data.feature_search`
        2. Impute default values, using the feature search dataset split, if the generator didn't provide default values
        3. Select the final features using `params.selector` using the enriched feature search dataset split
        4. Enrich `data.feature_evaluation` and `data.test` and calculate statistics on those two splits

        Raises:
            NoFeaturesFoundError: if no generator returned any candidate features
        """
        ...

class NoFeaturesFoundError(Exception): pass
