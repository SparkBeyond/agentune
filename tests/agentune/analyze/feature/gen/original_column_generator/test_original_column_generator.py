
from unittest.mock import MagicMock

import duckdb
import polars as pl

from agentune.analyze.feature.gen.original_column_generator.features import (
    OriginalBoolFeature,
    OriginalCategoricalFeature,
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


def test_original_columns_categorical_types() -> None:
    """Test that supported column types are properly included."""
    n_rows = 100
    df = pl.DataFrame({
        # Enum type - should be included
        'enum_col': (['red', 'blue', 'green'] * 34)[:n_rows],

        # String with low cardinality - should be included
        'low_card_string': (['cat', 'dog', 'bird'] * 34)[:n_rows],
    }).with_columns(
        pl.col('enum_col').cast(pl.Enum(['red', 'blue', 'green', 'yellow']))
    )

    feature_map = _generate_features(df)

    assert 'enum_col' in feature_map
    assert isinstance(feature_map['enum_col'], OriginalCategoricalFeature)

    assert 'low_card_string' in feature_map
    assert isinstance(feature_map['low_card_string'], OriginalCategoricalFeature)


def test_original_columns_skipped_types() -> None:
    """Test that unsupported column types are properly skipped."""
    from datetime import date

    n_rows = 100
    df = pl.DataFrame({
        # Temporal types - should be skipped
        'date_col': [date(2020, 1, 1), date(2020, 1, 2)] * 50,

        # Nested types - should be skipped
        'list_col': [[1, 2], [3, 4]] * 50,
        'struct_col': [{'a': 1}, {'a': 2}] * 50,

        # String with high cardinality - should be skipped (poor coverage)
        'high_card_string': [f'value_{i}' for i in range(n_rows)],

        # Valid numeric column for reference
        'valid_col': list(range(n_rows))
    })
    
    feature_map = _generate_features(df)
    
    # Temporal and nested types should be skipped
    assert 'date_col' not in feature_map
    assert 'list_col' not in feature_map
    assert 'struct_col' not in feature_map

    # String with poor coverage should be skipped
    assert 'high_card_string' not in feature_map

    # Valid numeric column should be included (for reference)
    assert 'valid_col' in feature_map
    assert isinstance(feature_map['valid_col'], OriginalIntFeature)


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


def test_original_columns_int_clipping() -> None:
    """Test that integer columns with values outside int64 range are clipped."""
    # Create a DataFrame with uint64 values that exceed int64 max
    int64_max = 2**63 - 1
    
    # Use uint64 which can hold values larger than int64_max
    df = pl.DataFrame({
        'uint64_col': pl.Series([0, 100, int64_max, int64_max + 1000, 2**64 - 1], dtype=pl.UInt64),
        'normal_int': pl.Series([0, 100, 1000, 5000, 10000], dtype=pl.Int32),
    })
    
    feature_map = _generate_features(df)
    
    # Both columns should be included as OriginalIntFeature
    assert 'uint64_col' in feature_map
    assert isinstance(feature_map['uint64_col'], OriginalIntFeature)
    assert 'normal_int' in feature_map
    assert isinstance(feature_map['normal_int'], OriginalIntFeature)
    
    # Verify that the features clip values correctly
    dataset = Dataset.from_polars(df)
    conn = duckdb.connect()
    
    # Test uint64_col clipping - values exceeding int64_max should be clipped
    uint64_result = feature_map['uint64_col'].compute_batch(dataset, conn)
    assert uint64_result.dtype == pl.Int64
    assert uint64_result[3] == int64_max  # Clipped from int64_max + 1000
    assert uint64_result[4] == int64_max  # Clipped from 2**64 - 1
    
    # Test normal_int - should be cast to int64
    normal_int_result = feature_map['normal_int'].compute_batch(dataset, conn)
    assert normal_int_result.dtype == pl.Int64


def test_categorical_columns() -> None:
    """Test enum and string categorical column handling."""
    # Enum: uses all schema-defined categories
    # String with good coverage: top K categories (no order guarantee)
    # String with poor coverage: skipped
    n_rows = 200
    
    # Create distinct categories for 'status' to test top-9 limit
    # 9 categories with 20 rows each (180 total)
    top_categories = [f'top_{i}' for i in range(9) for _ in range(20)]
    # 10 categories with 2 rows each (20 total)
    avg_categories = [f'low_{i}' for i in range(10) for _ in range(2)]
    status_col = (top_categories + avg_categories)
    
    df = pl.DataFrame({
        'color': (['red', 'blue', 'green', 'yellow'] * 50),
        'status': status_col,
        'high_cardinality': [f'value_{i}' for i in range(n_rows)]
    }).with_columns(
        pl.col('color').cast(pl.Enum(['red', 'blue', 'green', 'yellow']))
    )

    feature_map = _generate_features(df)

    # Enum: should include all schema-defined categories
    assert 'color' in feature_map
    assert isinstance(feature_map['color'], OriginalCategoricalFeature)
    assert feature_map['color'].categories == ('red', 'blue', 'green', 'yellow')
    
    # String with good coverage: should be truncated to top 9 categories
    assert 'status' in feature_map
    assert isinstance(feature_map['status'], OriginalCategoricalFeature)
    assert len(feature_map['status'].categories) == 9
    
    # Verify that the top categories are indeed the ones selected
    expected_top = {f'top_{i}' for i in range(9)}
    assert set(feature_map['status'].categories) == expected_top
    
    # String with poor coverage: should be skipped
    assert 'high_cardinality' not in feature_map
