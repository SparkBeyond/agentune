from __future__ import annotations

from abc import ABC, abstractmethod

from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.context.base import TablesWithContextDefinitions
from agentune.analyze.core.dataset import Dataset, DatasetSource
from agentune.analyze.feature.base import Classification, Feature, Regression, TargetKind
from agentune.analyze.feature.eval.base import FeatureEvaluator
from agentune.analyze.feature.eval.universal import (
    UniversalAsyncFeatureEvaluator,
    UniversalSyncFeatureEvaluator,
)
from agentune.analyze.feature.gen.base import FeatureGenerator
from agentune.analyze.feature.select.base import EnrichedFeatureSelector, FeatureSelector
from agentune.analyze.feature.stats import stats_calculators
from agentune.analyze.feature.stats.base import (
    FeatureWithFullStats,
)
from agentune.analyze.feature.stats.stats_calculators import (
    CombinedSyncFeatureStatsCalculator,
    CombinedSyncRelationshipStatsCalculator,
)
from agentune.analyze.run.base import RunContext
from agentune.analyze.run.ingest.sampling import SplitDuckdbTable

# TODO remove the context indexing into the preprocessing phase

@frozen
class FeatureSearchInputData:
    feature_search: Dataset # Small dataset for feature generators, held in memory
    feature_eval: DatasetSource 
    train: DatasetSource # Includes the feature_search and feature_eval datasets
    test: DatasetSource
    target_column: str
    contexts: TablesWithContextDefinitions

    def __attrs_post_init__(self) -> None:
        if self.feature_search.schema != self.train.schema:
            raise ValueError('Feature search dataset schema must match train dataset schema')
        if self.feature_search.schema != self.test.schema:
            raise ValueError('Feature search dataset schema must match test dataset schema')
        if self.target_column not in self.feature_search.schema.names:
            raise ValueError(f'Target column {self.target_column} not found')
        
    @staticmethod
    def from_split_table(split_table: SplitDuckdbTable, target_column: str,
                         contexts: TablesWithContextDefinitions,
                         conn: DuckDBPyConnection) -> FeatureSearchInputData:
        return FeatureSearchInputData(
            feature_search=split_table.feature_search().to_dataset(conn),
            train=split_table.train(), 
            test=split_table.test(),
            feature_eval=split_table.feature_eval(),
            target_column=target_column,
            contexts=contexts
        )

# TODO add user-specified params describing project, problem, etc. in freeform for LLM

@frozen
class FeatureSearchParams[TK: TargetKind]:
    generators: tuple[FeatureGenerator, ...]
    selector: FeatureSelector[Feature, TK] | EnrichedFeatureSelector
    # TODO declare a not-necessarily-sync version of these APIs, even if the only implementation right now is sync
    relationship_stats_calculator: CombinedSyncRelationshipStatsCalculator[TK]
    # Must always include at least one evaluator willing to handle every feature generated.
    # Normally this means including the two universal evaluators at the end of the list.
    # Evaluators are tried in the order in which they appear.
    evaluators: tuple[type[FeatureEvaluator], ...] = (UniversalSyncFeatureEvaluator, UniversalAsyncFeatureEvaluator)
    feature_stats_calculator: CombinedSyncFeatureStatsCalculator = stats_calculators.default_feature_stats_calculator

@frozen 
class RegressionFeatureSearchParams(FeatureSearchParams[Regression]):
    # Redeclare to set the default
    relationship_stats_calculator: CombinedSyncRelationshipStatsCalculator[Regression] = stats_calculators.default_regression_calculator

@frozen
class ClassificationFeatureSearchParams(FeatureSearchParams[Classification]):
    # Redeclare to set the default
    relationship_stats_calculator: CombinedSyncRelationshipStatsCalculator[Classification] = stats_calculators.default_classification_calculator

@frozen
class FeatureSearchResults[TK: TargetKind]:
    features_with_train_stats: tuple[FeatureWithFullStats[Feature, TK], ...]
    features_with_test_stats: tuple[FeatureWithFullStats[Feature, TK], ...]

    def __attrs_post_init__(self) -> None:
        if tuple(f.feature for f in self.features_with_train_stats) != tuple(f.feature for f in self.features_with_test_stats):
            raise ValueError('Features with train stats must match features with test stats')

    @property
    def features(self) -> tuple[Feature, ...]:
        return tuple(f.feature for f in self.features_with_test_stats)


class FeatureSearchRunner(ABC):
    # TODO the real one needs to be async but that's harder to implement so it's sync for now
    @abstractmethod
    def run[TK: TargetKind](self, context: RunContext, data: FeatureSearchInputData, 
                                  params: FeatureSearchParams[TK]) -> FeatureSearchResults[TK]: ...
