
from attrs import frozen

from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.schema import Schema
from agentune.analyze.feature.base import Feature, IntFeature
from agentune.analyze.feature.dedup_names import deduplicate_feature_names, deduplicate_strings
from agentune.analyze.join.base import JoinStrategy


@frozen
class FeatureForTesting(IntFeature):
    name: str
    description: str = ''
    technical_description: str = ''
    params: Schema = Schema(())
    secondary_tables: tuple[DuckdbTable, ...] = ()
    join_strategies: tuple[JoinStrategy, ...] = ()

    # Redeclare attributes with defaults
    default_for_missing: int = 0

def test_dedup_names() -> None:
    assert deduplicate_strings([]) == []
    assert deduplicate_strings(['a', 'b', 'c']) == ['a', 'b', 'c']
    assert deduplicate_strings(['a', 'b', 'c', 'a', 'b', 'a_']) == ['a', 'b', 'c', 'a_', 'b_', 'a__']

    def features_with_names(names: list[str]) -> list[Feature]:
        return [FeatureForTesting(name) for name in names]

    assert deduplicate_feature_names([]) == []
    assert deduplicate_feature_names(features_with_names(['a', 'b', 'c'])) == \
            features_with_names(['a', 'b', 'c'])
    assert deduplicate_feature_names(features_with_names(['a', 'b', 'c', 'a', 'b', 'a_'])) == \
           features_with_names(['a', 'b', 'c', 'a_', 'b_', 'a__'])

