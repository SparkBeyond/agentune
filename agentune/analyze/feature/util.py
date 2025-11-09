import functools
from collections.abc import Sequence

import attrs
import polars as pl

from agentune.analyze.core.dataset import Dataset, DatasetSource
from agentune.analyze.core.schema import Schema
from agentune.analyze.feature.base import Feature
from agentune.analyze.feature.dedup_names import deduplicate_feature_names


def _check_features_match_schema(schema: Schema, features: Sequence[Feature]) -> None:
    deduplicated_features = deduplicate_feature_names(features)
    feature_by_name = { feature.name: feature for feature in deduplicated_features}
    for name in feature_by_name:
        if name not in schema.names:
            raise ValueError(f'Feature name {name} not found in data schema')


def substitute_default_values(dataset: Dataset, features: Sequence[Feature]) -> Dataset:
    """Apply each feature's .substitute_default_values method to the corresponding column in the dataset.

    Assumes the feature names match the column names (after applying the deduplication logic);
    fails if a column isn't found for one of the features.

    Does not require all columns in the dataset to be changed; the list of features can be shorter
    than the amount of dataset columns.
    """
    _check_features_match_schema(dataset.schema, features)

    deduplicated_features = deduplicate_feature_names(features)
    feature_by_name = { feature.name: feature for feature in deduplicated_features}

    new_cols = { name: feature_by_name[name].substitute_defaults_batch(col) if name in feature_by_name else col
                 for name, col in dataset.data.to_dict().items() }
    new_df = pl.DataFrame(new_cols)
    return attrs.evolve(dataset, data=new_df)

def substitute_default_values_stream(dataset_source: DatasetSource, features: Sequence[Feature]) -> DatasetSource:
    """Apply each feature's .substitute_default_values to the corresponding column in each dataset in the stream.

    See substitute_default_values.
    """
    # Eagerly check column validity; otherwise it would only be checked when the dataset source is consumed
    _check_features_match_schema(dataset_source.schema, features)
    return dataset_source.map(dataset_source.schema, functools.partial(substitute_default_values, features=features))

