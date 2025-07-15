import itertools
import logging
from typing import cast

import polars as pl
from tests.agentune.analyze.flow.feature_search.toys import (
    ToyAsyncFeatureDescriber,
    ToyAsyncFeatureEvaluator,
    ToyAsyncFeatureGenerator,
    ToyAsyncFeatureSelector,
    ToyAsyncFeatureTransformer,
    ToySyncFeatureDescriber,
    ToySyncFeatureEvaluator,
    ToySyncFeatureGenerator,
    ToySyncFeatureSelector,
    ToySyncFeatureTransformer,
)

from agentune.analyze.core.dataset import Dataset
from agentune.analyze.feature.base import Feature, Regression
from agentune.analyze.feature.eval.base import FeatureEvaluator
from agentune.analyze.feature.stats.base import FeatureWithFullStats
from agentune.analyze.feature.stats.stats_calculators import (
    CombinedSyncFeatureStatsCalculator,
    default_regression_calculator,
)
from agentune.analyze.flow.feature_search.base import (
    FeatureSearchDatasets,
    RegressionFeatureSearchParams,
)
from agentune.analyze.flow.feature_search.simple import SimpleFeatureSearchFlow

_logger = logging.getLogger(__name__)


def test_simple_flow() -> None:
    df = pl.DataFrame(
        {
            'x': [1.0, 2.0, 3.0],
            'y': [4.0, 5.0, 6.0],
            'z': [7.0, 8.0, 9.0],
            'target': [10.0, 11.0, 12.0],
        }
    )
    datasets = FeatureSearchDatasets(
        Dataset.from_polars(df), 'target', []
    )
    flow = SimpleFeatureSearchFlow()
    results: list[tuple[FeatureWithFullStats[Feature, Regression], ...]] = []

    for transformer in [ToySyncFeatureTransformer(), ToyAsyncFeatureTransformer(), None]:
        for describer in [ToySyncFeatureDescriber(), ToyAsyncFeatureDescriber(), None]:
            for feature_stats_calculator in [CombinedSyncFeatureStatsCalculator()]:
                for relationship_stats_calculator in [default_regression_calculator]:
                    for selector in [ToySyncFeatureSelector(), ToyAsyncFeatureSelector()]:
                        params = RegressionFeatureSearchParams(
                            datasets,
                            [ToySyncFeatureGenerator(), ToyAsyncFeatureGenerator()],
                            transformer,
                            describer,
                            cast(list[type[FeatureEvaluator]], [ToySyncFeatureEvaluator, ToyAsyncFeatureEvaluator]),
                            feature_stats_calculator,
                            relationship_stats_calculator,
                            selector)
                        results.append(flow.run(params))

    for result, index in zip(results, itertools.count()):
        _logger.info(f'Result set {index}')
        for feature, findex in zip(result, itertools.count()):
            _logger.info(f'\t{findex}: {feature.feature.name}')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_simple_flow()
