from collections.abc import Sequence

import attrs

from agentune.analyze.feature.base import Feature


def deduplicate_strings(names: Sequence[str], existing: Sequence[str] = ()) -> list[str]:
    """Return a list of unique strings corresponding to the input by appending one or more underscores to each input item.

    Args:
        existing: values that may not be used in the output, even though they don't appear in the input.
    """
    seen = set(existing)
    output = []
    for name in names:
        new_name = name
        while new_name in seen:
            new_name = new_name + '_'
        seen.add(new_name)
        output.append(new_name)
    return output

def deduplicate_feature_names(features: Sequence[Feature], existing_names: Sequence[str] = ()) -> list[Feature]:
    """Change feature names by appending one or more underscores so that all features in the returned list have distinct names.

    This is idempotent; if the features' names are already distinct, the original features are returned.

    Args:
        existing_names: names that are taken (e.g. by other columns) and that features are not allowed to have.
    """
    return [attrs.evolve(feature, name=new_name) if new_name != feature.name else feature
            for feature, new_name in zip(features, deduplicate_strings([feature.name for feature in features], existing=existing_names), strict=False)]
