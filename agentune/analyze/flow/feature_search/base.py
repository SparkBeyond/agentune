from abc import ABC, abstractmethod
from collections.abc import Sequence

from attrs import frozen

from agentune.analyze.context.base import ContextDefinition
from agentune.analyze.core.dataset import Dataset, DatasetStreamSource
from agentune.analyze.feature.base import Feature, Regression
from agentune.analyze.feature.describe.base import FeatureDescriber
from agentune.analyze.feature.eval.base import FeatureEvaluator
from agentune.analyze.feature.gen.base import FeatureGenerator, FeatureTransformer
from agentune.analyze.feature.select.base import FeatureSelector
from agentune.analyze.feature.stats.base import (
    FeatureStatsCalculator,
    FeatureWithFullStats,
    RelationshipStatsCalculator,
)


@frozen 
class ContextSource:
    name: str # will name the relation we create
    source: DatasetStreamSource
    context_definitions: Sequence[ContextDefinition]

@frozen
class FeatureSearchDatasets:
    feature_search: Dataset # In memory! 
    target_col: str
    context_sources: Sequence[ContextSource]

@frozen
class RegressionFeatureSearchParams:
    datasets: FeatureSearchDatasets
    feature_generators: Sequence[FeatureGenerator]
    # We could have several transformers, but then we'd have to define which ones gets to transform which features,
    # whether a transformer can transform features emitted by another transformer, etc.
    feature_transformer: FeatureTransformer | None
    feature_describer: FeatureDescriber | None
    feature_evaluators: Sequence[type[FeatureEvaluator]]
    feature_stats_calculator: FeatureStatsCalculator[Feature]
    relationship_stats_calculator: RelationshipStatsCalculator[Feature, Regression]
    feature_selector: FeatureSelector[Feature, Regression]
    

class FeatureSearchFlow(ABC):
    @abstractmethod
    def run(self, params: RegressionFeatureSearchParams) -> tuple[FeatureWithFullStats[Feature, Regression], ...]: ...
