from collections.abc import Sequence

import attrs

from agentune.analyze.feature.base import Feature


def _deduplicate(names: Sequence[str]) -> list[str]:
    seen = set()
    output = []
    for name in names:
        new_name = name
        while new_name in seen:
            new_name = new_name + '_'
        seen.add(new_name)
        output.append(new_name)
    return output

def deduplicate_feature_names(features: Sequence[Feature]) -> list[Feature]:
    """Change feature names by appending one or more underscores so that all features in the returned list have distinct names."""
    return [attrs.evolve(feature, name=new_name) if new_name != feature.name else feature
            for feature, new_name in zip(features, _deduplicate([feature.name for feature in features]), strict=False)]
