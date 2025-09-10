import math

import polars as pl
from duckdb.duckdb import DuckDBPyConnection
from tests.agentune.analyze.run.feature_search.toys import ToySyncFeature

from agentune.analyze.core.dataset import Dataset, DatasetSource
from agentune.analyze.feature import util
from agentune.analyze.feature.dedup_names import deduplicate_feature_names
from agentune.analyze.feature.eval.universal import UniversalSyncFeatureEvaluator
from agentune.analyze.run.enrich.impl import EnrichRunnerImpl


async def test_substitute_default_values_stream(conn: DuckDBPyConnection) -> None:
    def jiggle(value: float) -> float | None:
        if value == 0.0:
            return math.nan
        elif value == 1.0:
            return math.inf
        elif value == 2.0:
            return -math.inf
        elif value == 3.0:
            return None
        else:
            return value

    df = pl.DataFrame({
        'x': [jiggle(float(x % 20)) for x in range(1, 1000)],
        'y': [jiggle(float(y % 17)) for y in range(1, 1000)],
        'z': [jiggle(float(z % 15)) for z in range(1, 1000)],
        'target': [int(t % 3) for t in range(1, 1000)],
    })
    input_dataset = Dataset.from_polars(df)

    evaluators = (UniversalSyncFeatureEvaluator,)
    features = [
        ToySyncFeature('x', 'y', 'x+y', '', ''),
        ToySyncFeature('x', 'z', 'x+z', '', ''),
        ToySyncFeature('y', 'z', 'x+z', '', ''), # duplicate (incorrect) name
    ]
    enriched = await EnrichRunnerImpl().run(features, input_dataset, evaluators, conn,
                                            keep_input_columns=input_dataset.schema.names)
    enriched_source = DatasetSource.from_datasets(enriched.schema, [enriched])

    assert enriched.schema.names != [feature.name for feature in features], 'Feature names were deduplicated'
    assert len(enriched.schema.names) == len(features) + len(input_dataset.schema.names)
    for col in enriched.data.iter_columns():
        if col.name != 'target':
            assert col.null_count() > 0, f'Column {col.name} has nulls'
            assert col.is_nan().sum() > 0, f'Column {col.name} has nans'
            assert math.inf in col, f'Column {col.name} has infs'

    substituted = util.substitute_default_values_stream(enriched_source, features)
    assert substituted.schema == enriched_source.schema
    substituted_dataset = substituted.to_dataset(conn)
    assert substituted_dataset.schema == substituted.schema

    for feature in deduplicate_feature_names(features):
        enriched_col = enriched.data[feature.name]
        assert feature.substitute_defaults_batch(enriched_col).null_count() == 0, 'Feature substitutes nulls'

    assert not substituted_dataset.data.equals(enriched.data), 'Data was changed'
    for col in substituted_dataset.data.iter_columns():
        if col.name in [feature.name for feature in deduplicate_feature_names(features)]:
            assert col.null_count() == 0, f'Column {col.name} no longer has nulls'
        elif col.name != 'target':
            assert col.null_count() > 0, f'Input column {col.name} still has nulls'

    for feature in deduplicate_feature_names(features):
        enriched_col = enriched.data[feature.name]
        substituted_col = substituted_dataset.data[feature.name]
        assert substituted_col.equals(feature.substitute_defaults_batch(enriched_col), check_names=True, check_dtypes=True), 'Substituted correctly'
