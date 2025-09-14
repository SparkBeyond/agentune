"""Tests for type detection utilities."""

import httpx
import polars as pl
import pytest

from agentune.analyze.core import types
from agentune.analyze.core.llm import LLMContext, LLMSpec
from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.feature.gen.insightful_text_generator.schema import Query
from agentune.analyze.feature.gen.insightful_text_generator.type_detector import (
    cast_to_categorical,
    decide_dtype,
    detect_bool_type,
    detect_categorical_type,
    detect_float_type,
    detect_int_type,
)


class TestBasicTypeDetectors:
    """Test individual type detector functions."""
    
    def test_is_bool_true_false(self) -> None:
        """Test boolean detection with true/false values."""
        data = pl.Series(['true', 'false', 'True', 'FALSE', 'True', 'false'])
        result = detect_bool_type(data, max_error_percentage=0.01)
        assert result == types.boolean
    
    def test_is_bool_yes_no(self) -> None:
        """Test boolean detection with yes/no values."""
        data = pl.Series(['yes', 'no', 'Yes', 'NO', 'y', 'n'])
        result = detect_bool_type(data, max_error_percentage=0.01)
        assert result == types.boolean
    
    def test_is_bool_numeric(self) -> None:
        """Test boolean detection with numeric 0/1 values."""
        data = pl.Series(['1', '0', '1', '0', '1'])
        result = detect_bool_type(data, max_error_percentage=0.01)
        assert result == types.boolean
    
    def test_is_bool_mixed_valid(self) -> None:
        """Test boolean detection with mixed valid boolean values."""
        data = pl.Series(['true', 'no', '1', 'false', 'yes', '0'])
        result = detect_bool_type(data, max_error_percentage=0.01)
        assert result == types.boolean
    
    def test_is_bool_with_errors_within_threshold(self) -> None:
        """Test boolean detection with errors within threshold."""
        # 90% valid boolean values, 10% invalid
        data = pl.Series(['true', 'false', 'yes', 'no', '1', '0', 'true', 'false', 'yes', 'invalid'])
        result = detect_bool_type(data, max_error_percentage=0.15)  # Allow 15% errors
        assert result == types.boolean
    
    def test_is_bool_with_errors_above_threshold(self) -> None:
        """Test boolean detection with errors above threshold."""
        # 50% valid, 50% invalid
        data = pl.Series(['true', 'false', 'invalid1', 'invalid2', 'invalid3'])
        result = detect_bool_type(data, max_error_percentage=0.01)
        assert result is None
    
    def test_is_bool_non_boolean_data(self) -> None:
        """Test boolean detection with non-boolean data."""
        data = pl.Series(['apple', 'banana', 'cherry'])
        result = detect_bool_type(data, max_error_percentage=0.01)
        assert result is None
    
    def test_is_int_valid_integers(self) -> None:
        """Test integer detection with valid integer strings."""
        data = pl.Series(['1', '2', '3', '100', '50', '-5', '0'])
        result = detect_int_type(data, max_error_percentage=0.01)
        assert result == types.int32
    
    def test_is_int_with_errors_within_threshold(self) -> None:
        """Test integer detection with errors within threshold."""
        # 90% valid integers, 10% invalid
        data = pl.Series(['1', '2', '3', '4', '5', '6', '7', '8', '9', 'not_int'])
        result = detect_int_type(data, max_error_percentage=0.15)
        assert result == types.int32
    
    def test_is_int_with_errors_above_threshold(self) -> None:
        """Test integer detection with errors above threshold."""
        data = pl.Series(['1', '2', 'not_int', 'also_not_int'])
        result = detect_int_type(data, max_error_percentage=0.01)
        assert result is None
    
    def test_is_int_float_strings(self) -> None:
        """Test integer detection with float strings (should fail)."""
        data = pl.Series(['1.5', '2.7', '3.14'])
        result = detect_int_type(data, max_error_percentage=0.01)
        assert result is None
    
    def test_is_float_valid_floats(self) -> None:
        """Test float detection with valid float strings."""
        data = pl.Series(['1.5', '2.7', '3.14', '100.5', '-2.3', '0.0'])
        result = detect_float_type(data, max_error_percentage=0.01)
        assert result == types.float64
    
    def test_is_float_integers_as_floats(self) -> None:
        """Test float detection with integer strings (should work)."""
        data = pl.Series(['1', '2', '3', '100'])
        result = detect_float_type(data, max_error_percentage=0.01)
        assert result == types.float64
    
    def test_is_float_mixed_int_float(self) -> None:
        """Test float detection with mixed integer and float strings."""
        data = pl.Series(['1', '2.5', '3', '4.7', '100'])
        result = detect_float_type(data, max_error_percentage=0.01)
        assert result == types.float64
    
    def test_is_float_with_errors_within_threshold(self) -> None:
        """Test float detection with errors within threshold."""
        data = pl.Series(['1.5', '2.7', '3.14', '4.2', '5.1', '6.8', '7.3', '8.9', '9.1', 'not_float'])
        result = detect_float_type(data, max_error_percentage=0.15)
        assert result == types.float64
    
    def test_is_float_non_numeric_data(self) -> None:
        """Test float detection with non-numeric data."""
        data = pl.Series(['apple', 'banana', 'cherry'])
        result = detect_float_type(data, max_error_percentage=0.01)
        assert result is None
    
    def test_is_categorical_valid_categories(self) -> None:
        """Test categorical detection with valid categories."""
        data = pl.Series(['red', 'blue', 'red', 'green', 'blue', 'red', 'yellow'] * 10)  # 70 items
        result = detect_categorical_type(data, max_categorical=5, min_threshold_percentage=0.05, min_coverage=0.8, max_chars=30)
        assert isinstance(result, types.EnumDtype)
        assert len(result.values) <= 5
    
    def test_is_categorical_insufficient_coverage(self) -> None:
        """Test categorical detection with insufficient coverage."""
        # Create data where top categories don't meet coverage threshold
        data = pl.Series(['a'] * 10 + ['b'] * 5 + [f'rare_{i}' for i in range(100)])  # 115 total
        result = detect_categorical_type(data, max_categorical=5, min_threshold_percentage=0.01, min_coverage=0.8, max_chars=30)
        assert result is None
    
    def test_is_categorical_below_threshold_percentage(self) -> None:
        """Test categorical detection with categories below threshold percentage."""
        # All categories appear only once (too rare)
        data = pl.Series([f'item_{i}' for i in range(100)])
        result = detect_categorical_type(data, max_categorical=5, min_threshold_percentage=0.05, min_coverage=0.8, max_chars=30)
        assert result is None
    
    def test_is_categorical_exactly_max_categories(self) -> None:
        """Test categorical detection with exactly max categories."""
        data = pl.Series(['a'] * 30 + ['b'] * 25 + ['c'] * 20 + ['d'] * 15 + ['e'] * 10)  # 100 total
        result = detect_categorical_type(data, max_categorical=4, min_threshold_percentage=0.05, min_coverage=0.8, max_chars=30)
        assert isinstance(result, types.EnumDtype)
        assert len(result.values) == 4  # 4 categories


class TestDecideDtype:
    """Test the main decide_dtype function."""
    
    def test_decide_dtype_boolean(self) -> None:
        """Test decide_dtype with boolean data."""
        query = Query('test_bool', 'Is this true?', types.string)
        data = pl.Series(['true', 'false', 'yes', 'no'])
        result = decide_dtype(query, data, max_categorical=10)
        assert result == types.boolean
    
    def test_decide_dtype_integer(self) -> None:
        """Test decide_dtype with integer data."""
        query = Query('test_int', 'What is the count?', types.string)
        data = pl.Series(['1', '2', '3', '100', '50'])
        result = decide_dtype(query, data, max_categorical=10)
        assert result == types.int32
    
    def test_decide_dtype_float(self) -> None:
        """Test decide_dtype with float data."""
        query = Query('test_float', 'What is the value?', types.string)
        data = pl.Series(['1.5', '2.7', '3.14', '100.5'])
        result = decide_dtype(query, data, max_categorical=10)
        assert result == types.float64
    
    def test_decide_dtype_mixed_numeric(self) -> None:
        """Test decide_dtype with mixed integer and float data."""
        query = Query('test_mixed', 'What is the amount?', types.string)
        data = pl.Series(['1', '2.5', '3', '4.7', '100'])
        result = decide_dtype(query, data, max_categorical=10)
        assert result == types.float64
    
    def test_decide_dtype_categorical(self) -> None:
        """Test decide_dtype with categorical data."""
        query = Query('test_cat', 'What is the color?', types.string)
        data = pl.Series(['red', 'blue', 'red', 'green', 'blue', 'red'] * 10)
        result = decide_dtype(query, data, max_categorical=5)
        assert isinstance(result, types.EnumDtype)
    
    def test_decide_dtype_falls_back_to_string(self) -> None:
        """Test decide_dtype falls back to string for low-coverage data."""
        query = Query('test_string', 'What is the description?', types.string)
        # Create data with many unique values (low coverage)
        data = pl.Series([f'description_{i}' for i in range(100)])
        result = decide_dtype(query, data, max_categorical=5, min_categorical_coverage=0.8)
        assert result == types.string
    
    def test_decide_dtype_empty_data(self) -> None:
        """Test decide_dtype with empty data."""
        query = Query('test_empty', 'What is the value?', types.string)
        data = pl.Series([], dtype=pl.String)
        result = decide_dtype(query, data, max_categorical=10)
        assert result == types.string
    
    def test_decide_dtype_priority_order(self) -> None:
        """Test that decide_dtype respects type detection priority order."""
        query = Query('test_priority', 'What is the value?', types.string)
        
        # Boolean should take priority over categorical
        bool_data = pl.Series(['true', 'false', 'true', 'false'])
        result = decide_dtype(query, bool_data, max_categorical=10)
        assert result == types.boolean
        
        # Integer should take priority over float and categorical
        int_data = pl.Series(['1', '2', '3', '1', '2', '3'])
        result = decide_dtype(query, int_data, max_categorical=10)
        assert result == types.int32


@pytest.fixture
async def real_llm_with_spec(httpx_async_client: httpx.AsyncClient) -> LLMWithSpec:
    """Create a real LLM for end-to-end testing."""
    llm_context = LLMContext(httpx_async_client)
    llm_spec = LLMSpec('openai', 'o3')  # Use a smart model for testing
    llm_with_spec = LLMWithSpec(
        llm=llm_context.from_spec(llm_spec),
        spec=llm_spec
    )
    return llm_with_spec


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cast_to_categorical_with_real_llm(real_llm_with_spec: LLMWithSpec) -> None:
    """Integration test for cast_to_categorical with real LLM."""
    original_query = Query(
        name='products_mentioned',
        query_text='What products are mentioned in the conversation?',
        return_type=types.string
    )
    
    # Create problematic data with inconsistent naming
    problematic_data = pl.Series([
        'headphones', 'In-Ear headphones', 'headphone', 'gaming headphones',
        'laptop', 'gaming laptop', 'MacBook', 'macbook pro', 'laptop computer',
        'mouse', 'gaming mouse', 'wireless mouse', 'computer mouse',
        'keyboard', 'mechanical keyboard', 'gaming keyboard',
        'headphones', 'laptop', 'mouse',  # More frequent items
        'headphones', 'laptop', 'mouse', 'keyboard'
    ])
    
    updated_query = await cast_to_categorical(
        original_query, problematic_data, max_categorical=5, llm=real_llm_with_spec
    )
    
    # Validate the results
    assert updated_query.name != original_query.name or updated_query.query_text != original_query.query_text
    assert isinstance(updated_query.return_type, types.EnumDtype)
    assert len(updated_query.return_type.values) <= 5
    assert all(isinstance(category, str) for category in updated_query.return_type.values)
    assert len(updated_query.name) > 0
    assert len(updated_query.query_text) > 0
    
    # Ensure no "others" category is included (it should be handled by the system)
    assert 'others' not in [v.lower() for v in updated_query.return_type.values]


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_type_detection_with_none_values(self) -> None:
        """Test type detection handles None/null values properly."""
        # The functions expect non-null data, but test resilience
        query = Query('test_with_nulls', 'What is the value?', types.string)
        data = pl.Series(['1', '2', '3', None, '4'])
        
        # Filter out nulls as the actual implementation would do
        non_null_data = data.drop_nulls()
        result = decide_dtype(query, non_null_data, max_categorical=10)
        assert result == types.int32
    
    def test_single_value_data(self) -> None:
        """Test type detection with single value."""
        query = Query('test_single', 'What is the value?', types.string)
        data = pl.Series(['true'])
        result = decide_dtype(query, data, max_categorical=10)
        assert result == types.boolean
    
    def test_very_large_categorical_data(self) -> None:
        """Test categorical detection with large number of categories."""
        query = Query('test_large_cat', 'What is the category?', types.string)
        # Create data with many categories but clear winners
        data = pl.Series(
            ['category_a'] * 1000
            + ['category_b'] * 800
            + ['category_c'] * 600
            + [f'rare_{i}' for i in range(100)]
        )
        result = decide_dtype(query, data, max_categorical=5, min_categorical_coverage=0.8)
        assert isinstance(result, types.EnumDtype)
    
    def test_all_same_value(self) -> None:
        """Test type detection when all values are the same."""
        query = Query('test_same', 'What is the value?', types.string)
        data = pl.Series(['constant'] * 100)
        result = decide_dtype(query, data, max_categorical=5)
        assert isinstance(result, types.EnumDtype)
        assert 'constant' in result.values
