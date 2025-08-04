
from attrs import frozen

from agentune.analyze.context.base import ContextDefinition
from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.schema import Schema
from agentune.analyze.feature.base import Feature, IntFeature
from agentune.analyze.feature.dedup_names import _deduplicate, deduplicate_feature_names


@frozen
class TestFeature(IntFeature):
    name: str
    code: str = ''
    description: str = ''
    params: Schema = Schema(())
    context_tables: tuple[DuckdbTable, ...] = ()
    context_objects: tuple[ContextDefinition, ...] = ()

def test_dedup_names() -> None:
    assert _deduplicate([]) == []
    assert _deduplicate(['a', 'b', 'c']) == ['a', 'b', 'c']
    assert _deduplicate(['a', 'b', 'c', 'a', 'b', 'a_']) == ['a', 'b', 'c', 'a_', 'b_', 'a__']

    def features_with_names(names: list[str]) -> list[Feature]:
        return [TestFeature(name) for name in names]

    assert deduplicate_feature_names([]) == []
    assert deduplicate_feature_names(features_with_names(['a', 'b', 'c'])) == \
            features_with_names(['a', 'b', 'c'])
    assert deduplicate_feature_names(features_with_names(['a', 'b', 'c', 'a', 'b', 'a_'])) == \
           features_with_names(['a', 'b', 'c', 'a_', 'b_', 'a__'])

