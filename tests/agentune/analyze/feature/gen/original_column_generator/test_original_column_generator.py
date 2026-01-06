
from unittest.mock import MagicMock

import duckdb
import polars as pl

from agentune.analyze.feature.gen.original_column_generator.features import (
    OriginalBoolFeature,
    OriginalFloatFeature,
    OriginalIntFeature,
)
from agentune.analyze.feature.gen.original_column_generator.original_column_generator import (
    OriginalColumnsGenerator,
)
from agentune.core.dataset import Dataset


def _generate_features(df: pl.DataFrame) -> dict:
    """Helper to generate features from a DataFrame."""
    dataset = Dataset.from_polars(df)
    generator = OriginalColumnsGenerator()
    features = list(generator.generate(
        dataset,
        MagicMock(),
        MagicMock(),
        duckdb.connect()
    ))
    return {f.feature.name: f.feature for f in features}


def test_original_columns_generator_basics() -> None:
    # Setup data
    df = pl.DataFrame({
        'int32_col': [1, 2, 3, 4],
        'float_col': [1.1, 2.2, 3.3, 4.4],
        'bool_col': [True, False, True, False],
        'constant_col': [1, 1, 1, 1],  # Should be skipped
    }).cast({
        'int32_col': pl.Int32,
        'float_col': pl.Float64,
        'bool_col': pl.Boolean,
        'constant_col': pl.Int32
    })
    
    feature_map = _generate_features(df)
    
    assert 'int32_col' in feature_map
    assert isinstance(feature_map['int32_col'], OriginalIntFeature)
    
    assert 'float_col' in feature_map
    assert isinstance(feature_map['float_col'], OriginalFloatFeature)
    
    assert 'bool_col' in feature_map
    assert isinstance(feature_map['bool_col'], OriginalBoolFeature)
    
    assert 'constant_col' not in feature_map


def test_original_columns_skipping_types() -> None:
    from datetime import date
    
    df = pl.DataFrame({
        'date_col': [date(2020, 1, 1), date(2020, 1, 2)],
        'list_col': [[1, 2], [3, 4]],
        'struct_col': [{'a': 1}, {'a': 2}],
        'valid_col': [1, 2]
    })
    
    feature_map = _generate_features(df)
    
    assert 'date_col' not in feature_map
    assert 'list_col' not in feature_map
    assert 'struct_col' not in feature_map
    assert 'valid_col' in feature_map


def test_original_columns_with_special_values() -> None:
    """Test that features handle missing values and infinities correctly."""
    
    df = pl.DataFrame({
        'int_with_nulls': [1, 2, None, 4, 5],
        'float_with_nulls': [1.1, None, 3.3, 4.4, 5.5],
        'float_with_inf': [1.0, 2.0, float('inf'), 4.0, 5.0],
        'float_with_neg_inf': [1.0, float('-inf'), 3.0, 4.0, 5.0],
        'float_with_nan': [1.0, 2.0, float('nan'), 4.0, 5.0],
        'float_with_all': [1.0, None, float('inf'), float('-inf'), float('nan')],
    })
    
    feature_map = _generate_features(df)
    
    # All columns should be generated (not skipped due to special values)
    assert 'int_with_nulls' in feature_map
    assert 'float_with_nulls' in feature_map
    assert 'float_with_inf' in feature_map
    assert 'float_with_neg_inf' in feature_map
    assert 'float_with_nan' in feature_map
    assert 'float_with_all' in feature_map
    
    # Verify correct feature types
    assert isinstance(feature_map['int_with_nulls'], OriginalIntFeature)
    assert isinstance(feature_map['float_with_nulls'], OriginalFloatFeature)
    assert isinstance(feature_map['float_with_inf'], OriginalFloatFeature)
    
    # Verify features have default values defined (even though has_good_defaults=False)
    float_feature = feature_map['float_with_all']
    assert hasattr(float_feature, 'default_for_missing')
    assert hasattr(float_feature, 'default_for_nan')
    assert hasattr(float_feature, 'default_for_infinity')
    assert hasattr(float_feature, 'default_for_neg_infinity')
