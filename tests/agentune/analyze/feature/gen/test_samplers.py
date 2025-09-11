"""Tests for data sampling utilities."""

import polars as pl
import pytest

from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.core.types import EnumDtype, boolean, float64, int32, string
from agentune.analyze.feature.gen.insightful_text_generator.sampling.base import (
    RandomSampler,
)
from agentune.analyze.feature.gen.insightful_text_generator.sampling.samplers import (
    BalancedClassSampler,
    BalancedNumericSampler,
    ProportionalClassSampler,
    ProportionalNumericSampler,
    validate_field_exists_and_matches_schema,
    validate_field_is_numeric,
)


def create_test_dataset(data_dict: dict[str, list], schema_fields: list[Field]) -> Dataset:
    """Helper to create test datasets."""
    schema = Schema(tuple(schema_fields))
    data = pl.DataFrame(data_dict)
    return Dataset(schema=schema, data=data)


class TestRandomSampler:
    """Test RandomSampler functionality and edge cases."""
    
    def test_basic_random_sampling(self) -> None:
        """Test basic random sampling functionality."""
        sampler = RandomSampler()
        
        # Create test dataset
        data_dict: dict[str, list] = {'id': list(range(100)), 'value': [f'item_{i}' for i in range(100)]}
        schema_fields = [Field('id', int32), Field('value', string)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        # Sample data
        result = sampler.sample(dataset, sample_size=20, random_seed=42)
        
        # Validate result
        assert result.data.height == 20
        assert result.schema == dataset.schema
        assert set(result.data.columns) == set(dataset.data.columns)
    
    def test_random_sampling_reproducibility(self) -> None:
        """Test that random sampling is reproducible with same seed."""
        sampler = RandomSampler()
        
        data_dict: dict[str, list] = {'id': list(range(50)), 'value': [f'item_{i}' for i in range(50)]}
        schema_fields = [Field('id', int32), Field('value', string)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        # Sample with same seed multiple times
        result1 = sampler.sample(dataset, sample_size=10, random_seed=123)
        result2 = sampler.sample(dataset, sample_size=10, random_seed=123)
        
        # Results should be identical
        assert result1.data.equals(result2.data)
    
    def test_random_sampling_different_seeds(self) -> None:
        """Test that different seeds produce different results."""
        sampler = RandomSampler()
        
        data_dict: dict[str, list] = {'id': list(range(100)), 'value': [f'item_{i}' for i in range(100)]}
        schema_fields = [Field('id', int32), Field('value', string)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        # Sample with different seeds
        result1 = sampler.sample(dataset, sample_size=30, random_seed=42)
        result2 = sampler.sample(dataset, sample_size=30, random_seed=123)
        
        # Results should be different (with high probability)
        assert not result1.data.equals(result2.data)
    
    def test_random_sampling_edge_cases(self) -> None:
        """Test edge cases for random sampling."""
        sampler = RandomSampler()
        
        # Single row dataset
        data_dict: dict[str, list] = {'id': [1], 'value': ['single']}
        schema_fields = [Field('id', int32), Field('value', string)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        result = sampler.sample(dataset, sample_size=1, random_seed=42)
        assert result.data.height == 1
        assert result.data.equals(dataset.data)
    
    def test_random_sampling_validation_errors(self) -> None:
        """Test validation errors in random sampling."""
        sampler = RandomSampler()
        
        data_dict: dict[str, list] = {'id': list(range(10)), 'value': [f'item_{i}' for i in range(10)]}
        schema_fields = [Field('id', int32), Field('value', string)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        # Empty dataset
        empty_data = pl.DataFrame({'id': [], 'value': []}, schema={'id': pl.Int32, 'value': pl.String})
        empty_dataset = Dataset(schema=Schema(tuple(schema_fields)), data=empty_data)
        with pytest.raises(ValueError, match='Cannot sample from empty dataset'):
            sampler.sample(empty_dataset, sample_size=5)
        
        # Negative sample size
        with pytest.raises(ValueError, match='Sample size must be positive'):
            sampler.sample(dataset, sample_size=-1)
        
        # Zero sample size
        with pytest.raises(ValueError, match='Sample size must be positive'):
            sampler.sample(dataset, sample_size=0)
        
        # Sample size exceeds dataset size
        with pytest.raises(ValueError, match='Sample size 20 exceeds dataset size 10'):
            sampler.sample(dataset, sample_size=20)


class TestValidateFieldFunctions:
    """Test the field validation utility functions."""
    
    def test_validate_field_exists_and_matches_schema_valid(self) -> None:
        """Test validation passes for valid fields that exist in schema."""
        data_dict: dict[str, list] = {'id': [1, 2, 3], 'category': ['A', 'B', 'A']}
        enum_dtype = EnumDtype('A', 'B', 'C')
        schema_fields = [Field('id', int32), Field('category', enum_dtype)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        target_field = Field('category', enum_dtype)
        
        # Should not raise
        validate_field_exists_and_matches_schema(dataset, target_field)
    
    def test_validate_field_exists_and_matches_schema_not_found(self) -> None:
        """Test validation fails when field not in dataset."""
        data_dict: dict[str, list] = {'id': [1, 2, 3]}
        schema_fields = [Field('id', int32)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        target_field = Field('missing', string)
        with pytest.raises(ValueError, match="Target field 'missing' not found in dataset schema"):
            validate_field_exists_and_matches_schema(dataset, target_field)
    
    def test_validate_field_exists_and_matches_schema_dtype_mismatch(self) -> None:
        """Test validation fails when dtype doesn't match."""
        data_dict: dict[str, list] = {'id': [1, 2, 3], 'value': ['A', 'B', 'C']}
        schema_fields = [Field('id', int32), Field('value', string)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        # Wrong dtype
        target_field = Field('value', int32)  # Dataset has string, but field expects int32
        with pytest.raises(ValueError, match='Target field dtype mismatch'):
            validate_field_exists_and_matches_schema(dataset, target_field)
    
    def test_validate_field_is_numeric_valid(self) -> None:
        """Test validation passes for numeric fields."""
        numeric_field = Field('score', float64)
        # Should not raise
        validate_field_is_numeric(numeric_field)
        
        int_field = Field('count', int32)
        # Should not raise
        validate_field_is_numeric(int_field)
    
    def test_validate_field_is_numeric_invalid(self) -> None:
        """Test validation fails for non-numeric fields."""
        string_field = Field('name', string)
        with pytest.raises(ValueError, match="Field 'name' must be numeric, got"):
            validate_field_is_numeric(string_field)
        
        enum_field = Field('category', EnumDtype('A', 'B'))
        with pytest.raises(ValueError, match="Field 'category' must be numeric, got"):
            validate_field_is_numeric(enum_field)
    

class TestProportionalClassSampler:
    """Test ProportionalClassSampler functionality and edge cases."""
    
    def test_proportional_class_basic(self) -> None:
        """Test basic proportional class sampling."""
        enum_dtype = EnumDtype('red', 'blue', 'green')
        target_field = Field('color', enum_dtype)
        sampler = ProportionalClassSampler(target_field=target_field)
        
        # Create imbalanced dataset: 60% red, 30% blue, 10% green
        data_dict: dict[str, list] = {
            'id': list(range(100)),
            'color': (['red'] * 60) + (['blue'] * 30) + (['green'] * 10)
        }
        schema_fields = [Field('id', int32), Field('color', enum_dtype)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        result = sampler.sample(dataset, sample_size=30, random_seed=42)
        
        # Check proportions are maintained
        color_counts = result.data['color'].value_counts().sort('color')
        total_sampled = result.data.height
        
        # Should roughly maintain 60:30:10 ratio
        red_count = color_counts.filter(pl.col('color') == 'red')['count'][0]
        blue_count = color_counts.filter(pl.col('color') == 'blue')['count'][0]
        green_count = color_counts.filter(pl.col('color') == 'green')['count'][0]
        
        # Allow some tolerance due to rounding
        assert red_count >= 15  # Should be ~18 (60% of 30)
        assert blue_count >= 7   # Should be ~9 (30% of 30)
        assert green_count >= 1  # Should be ~3 (10% of 30)
        assert red_count + blue_count + green_count == total_sampled
    
    def test_proportional_class_equal_distribution(self) -> None:
        """Test proportional sampling with equal class distribution."""
        bool_field = Field('flag', boolean)
        sampler = ProportionalClassSampler(target_field=bool_field)
        
        # Equal distribution
        data_dict: dict[str, list] = {
            'id': list(range(100)),
            'flag': ([True] * 50) + ([False] * 50)
        }
        schema_fields = [Field('id', int32), Field('flag', boolean)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        result = sampler.sample(dataset, sample_size=20, random_seed=42)
        
        flag_counts = result.data['flag'].value_counts()
        # Handle boolean filtering properly
        true_rows = flag_counts.filter(pl.col('flag'))
        false_rows = flag_counts.filter(~pl.col('flag'))
        true_count = true_rows['count'][0] if true_rows.height > 0 else 0
        false_count = false_rows['count'][0] if false_rows.height > 0 else 0
        
        # Should be approximately equal
        assert abs(true_count - false_count) <= 1
        assert true_count + false_count == 20
    
    def test_proportional_class_single_class(self) -> None:
        """Test proportional sampling with only one class."""
        enum_dtype = EnumDtype('only_value')
        target_field = Field('category', enum_dtype)
        sampler = ProportionalClassSampler(target_field=target_field)
        
        data_dict: dict[str, list] = {
            'id': list(range(50)),
            'category': ['only_value'] * 50
        }
        schema_fields = [Field('id', int32), Field('category', enum_dtype)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        result = sampler.sample(dataset, sample_size=10, random_seed=42)
        
        assert result.data.height == 10
        assert all(result.data['category'] == 'only_value')
    
    def test_proportional_class_small_classes(self) -> None:
        """Test proportional sampling when some classes have very few samples."""
        enum_dtype = EnumDtype('common', 'rare1', 'rare2')
        target_field = Field('type', enum_dtype)
        sampler = ProportionalClassSampler(target_field=target_field)
        
        # 90% common, 5% each rare
        data_dict: dict[str, list] = {
            'id': list(range(100)),
            'type': (['common'] * 90) + (['rare1'] * 5) + (['rare2'] * 5)
        }
        schema_fields = [Field('id', int32), Field('type', enum_dtype)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        result = sampler.sample(dataset, sample_size=20, random_seed=42)
        
        # Even rare classes should get at least 1 sample due to max(1, ...) in the code
        type_counts = result.data['type'].value_counts()
        unique_types = set(type_counts['type'].to_list())
        
        # All three types should be represented
        assert 'common' in unique_types
        assert 'rare1' in unique_types
        assert 'rare2' in unique_types
    

class TestProportionalNumericSampler:
    """Test ProportionalNumericSampler functionality and edge cases."""
    
    def test_proportional_numeric_basic(self) -> None:
        """Test basic proportional numeric sampling."""
        target_field = Field('score', float64)
        sampler = ProportionalNumericSampler(num_bins=4, target_field=target_field)
        
        # Create dataset with known distribution
        data_dict: dict[str, list] = {
            'id': list(range(100)),
            'score': [i / 10.0 for i in range(100)]  # 0.0 to 9.9
        }
        schema_fields = [Field('id', int32), Field('score', float64)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        result = sampler.sample(dataset, sample_size=20, random_seed=42)
        
        # Check that we have representation from different strata
        scores = result.data['score'].to_list()
        min_score, max_score = min(scores), max(scores)
        
        # Should span a reasonable range
        assert max_score > min_score
        assert result.data.height == 20
    
    def test_proportional_numeric_equal_strata(self) -> None:
        """Test that strata get proportional representation."""
        target_field = Field('value', int32)
        sampler = ProportionalNumericSampler(num_bins=3, target_field=target_field)
        
        # Values from 0 to 299 (300 total)
        data_dict: dict[str, list] = {
            'id': list(range(300)),
            'value': list(range(300))
        }
        schema_fields = [Field('id', int32), Field('value', int32)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        result = sampler.sample(dataset, sample_size=30, random_seed=42)
        
        # Each stratum should get 10 samples (30 / 3)
        # Check distribution across ranges
        values = result.data['value'].to_list()
        low_count = sum(1 for v in values if v < 100)
        mid_count = sum(1 for v in values if 100 <= v < 200)
        high_count = sum(1 for v in values if v >= 200)
        
        # Each range should have approximately equal representation
        assert low_count >= 8 and low_count <= 12  # Around 10, with tolerance
        assert mid_count >= 8 and mid_count <= 12
        assert high_count >= 8 and high_count <= 12
        assert low_count + mid_count + high_count == 30
    
    def test_proportional_numeric_all_same_value(self) -> None:
        """Test proportional sampling when all values are the same."""
        target_field = Field('constant', float64)
        sampler = ProportionalNumericSampler(num_bins=3, target_field=target_field)
        
        data_dict: dict[str, list] = {
            'id': list(range(50)),
            'constant': [5.0] * 50
        }
        schema_fields = [Field('id', int32), Field('constant', float64)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        result = sampler.sample(dataset, sample_size=15, random_seed=42)
        
        # Should still work and return requested number of samples
        assert result.data.height == 15
        assert all(result.data['constant'] == 5.0)
    
    def test_proportional_numeric_min_strata_validation(self) -> None:
        """Test validation of minimum number of strata."""
        target_field = Field('value', float64)
        
        # Should raise error for num_bins < 2
        with pytest.raises(ValueError, match='Number of bins must be at least 2'):
            ProportionalNumericSampler(num_bins=1, target_field=target_field)
        
        with pytest.raises(ValueError, match='Number of bins must be at least 2'):
            ProportionalNumericSampler(num_bins=0, target_field=target_field)
    
    def test_proportional_numeric_non_numeric_field(self) -> None:
        """Test validation error for non-numeric target field."""
        target_field = Field('category', string)
        sampler = ProportionalNumericSampler(num_bins=3, target_field=target_field)
        
        data_dict: dict[str, list] = {'id': [1, 2, 3], 'category': ['A', 'B', 'C']}
        schema_fields = [Field('id', int32), Field('category', string)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        with pytest.raises(ValueError, match="Field 'category' must be numeric"):
            sampler.sample(dataset, sample_size=2)


class TestBalancedClassSampler:
    """Test BalancedClassSampler functionality and edge cases."""
    
    def test_balanced_class_basic(self) -> None:
        """Test basic balanced class sampling."""
        enum_dtype = EnumDtype('red', 'blue', 'green')
        target_field = Field('color', enum_dtype)
        sampler = BalancedClassSampler(target_field=target_field)
        
        # Create heavily imbalanced dataset
        data_dict: dict[str, list] = {
            'id': list(range(130)),
            'color': (['red'] * 100) + (['blue'] * 20) + (['green'] * 10)
        }
        schema_fields = [Field('id', int32), Field('color', enum_dtype)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        result = sampler.sample(dataset, sample_size=30, random_seed=42)
        
        # Each class should get 10 samples (30 / 3 classes)
        color_counts = result.data['color'].value_counts().sort('color')
        
        red_count = color_counts.filter(pl.col('color') == 'red')['count'][0]
        blue_count = color_counts.filter(pl.col('color') == 'blue')['count'][0]
        green_count = color_counts.filter(pl.col('color') == 'green')['count'][0]
        
        assert red_count == 10
        assert blue_count == 10
        assert green_count == 10
    
    def test_balanced_class_limited_by_smallest_class(self) -> None:
        """Test balanced sampling when smallest class limits the sample size."""
        bool_field = Field('flag', boolean)
        sampler = BalancedClassSampler(target_field=bool_field)
        
        # Very imbalanced: 95 True, 5 False
        data_dict: dict[str, list] = {
            'id': list(range(100)),
            'flag': ([True] * 95) + ([False] * 5)
        }
        schema_fields = [Field('id', int32), Field('flag', boolean)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        result = sampler.sample(dataset, sample_size=20, random_seed=42)
        
        # Each class should get min(10, available) samples
        flag_counts = result.data['flag'].value_counts()
        
        # Handle boolean filtering properly - get counts for True and False values
        true_rows = flag_counts.filter(pl.col('flag'))  # Filter for True values
        false_rows = flag_counts.filter(~pl.col('flag'))  # Filter for False values
        
        true_count = true_rows['count'][0] if true_rows.height > 0 else 0
        false_count = false_rows['count'][0] if false_rows.height > 0 else 0
        
        # False class has only 5 samples, so both should get 5
        assert true_count == 10  # 20 // 2 = 10, but True class has enough
        assert false_count == 5   # Limited by available samples
    
    def test_balanced_class_single_class(self) -> None:
        """Test balanced sampling with only one class."""
        enum_dtype = EnumDtype('only')
        target_field = Field('type', enum_dtype)
        sampler = BalancedClassSampler(target_field=target_field)
        
        data_dict: dict[str, list] = {
            'id': list(range(30)),
            'type': ['only'] * 30
        }
        schema_fields = [Field('id', int32), Field('type', enum_dtype)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        result = sampler.sample(dataset, sample_size=15, random_seed=42)
        
        assert result.data.height == 15
        assert all(result.data['type'] == 'only')
    
    def test_balanced_class_empty_classes(self) -> None:
        """Test balanced sampling behavior with no classes."""
        enum_dtype = EnumDtype('A', 'B')
        target_field = Field('category', enum_dtype)
        sampler = BalancedClassSampler(target_field=target_field)
        
        # Empty dataset
        data_dict: dict[str, list] = {'id': [], 'category': []}
        schema = Schema((Field('id', int32), Field('category', enum_dtype)))
        empty_data = pl.DataFrame(data_dict, schema={'id': pl.Int32, 'category': pl.Enum(['A', 'B'])})
        dataset = Dataset(schema=schema, data=empty_data)
        
        with pytest.raises(ValueError, match='Cannot sample from empty dataset'):
            sampler.sample(dataset, sample_size=5)


class TestBalancedNumericSampler:
    """Test BalancedNumericSampler functionality and edge cases."""
    
    def test_balanced_numeric_basic(self) -> None:
        """Test basic balanced numeric sampling."""
        target_field = Field('score', float64)
        sampler = BalancedNumericSampler(num_bins=4, target_field=target_field)
        
        # Create dataset with values from 0 to 100
        data_dict: dict[str, list] = {
            'id': list(range(200)),
            'score': [i / 2.0 for i in range(200)]  # 0.0 to 99.5
        }
        schema_fields = [Field('id', int32), Field('score', float64)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        result = sampler.sample(dataset, sample_size=40, random_seed=42)
        
        # Each bin should get 10 samples (40 / 4)
        scores = result.data['score'].to_list()
        
        # Check distribution across bins: [0,25), [25,50), [50,75), [75,100)
        bin1_count = sum(1 for s in scores if 0 <= s < 25)
        bin2_count = sum(1 for s in scores if 25 <= s < 50)
        bin3_count = sum(1 for s in scores if 50 <= s < 75)
        bin4_count = sum(1 for s in scores if 75 <= s < 100)
        
        # Each bin should have approximately equal representation
        assert bin1_count >= 8 and bin1_count <= 12
        assert bin2_count >= 8 and bin2_count <= 12
        assert bin3_count >= 8 and bin3_count <= 12
        assert bin4_count >= 8 and bin4_count <= 12
        assert bin1_count + bin2_count + bin3_count + bin4_count == 40
    
    def test_balanced_numeric_all_same_value(self) -> None:
        """Test balanced sampling when all values are identical."""
        target_field = Field('constant', int32)
        sampler = BalancedNumericSampler(num_bins=3, target_field=target_field)
        
        data_dict: dict[str, list] = {
            'id': list(range(60)),
            'constant': [42] * 60
        }
        schema_fields = [Field('id', int32), Field('constant', int32)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        result = sampler.sample(dataset, sample_size=20, random_seed=42)
        
        # Should fall back to random sampling since min == max
        assert result.data.height == 20
        assert all(result.data['constant'] == 42)
    
    def test_balanced_numeric_edge_bins(self) -> None:
        """Test that edge cases in binning are handled correctly."""
        target_field = Field('value', float64)
        sampler = BalancedNumericSampler(num_bins=2, target_field=target_field)
        
        # Values: 0, 10, 20, ..., 90 (min=0, max=90, bin_width=45)
        # First bin: [0, 45], Second bin: (45, 90]
        data_dict: dict[str, list] = {
            'id': list(range(10)),
            'value': [float(i * 10) for i in range(10)]  # 0, 10, 20, ..., 90
        }
        schema_fields = [Field('id', int32), Field('value', float64)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        result = sampler.sample(dataset, sample_size=6, random_seed=42)
        
        values = result.data['value'].to_list()
        
        # First bin: 0 <= value <= 45 (includes 0, 10, 20, 30, 40)
        # Second bin: 45 < value <= 90 (includes 50, 60, 70, 80, 90)
        bin1_count = sum(1 for v in values if v <= 45)
        bin2_count = sum(1 for v in values if v > 45)
        
        # Each bin should get 3 samples (6 / 2)
        assert bin1_count == 3
        assert bin2_count == 3
    
    def test_balanced_numeric_min_bins_validation(self) -> None:
        """Test validation of minimum number of bins."""
        target_field = Field('value', float64)
        
        # Should raise error for num_bins < 2
        with pytest.raises(ValueError, match='Number of bins must be at least 2'):
            BalancedNumericSampler(num_bins=1, target_field=target_field)
        
        with pytest.raises(ValueError, match='Number of bins must be at least 2'):
            BalancedNumericSampler(num_bins=0, target_field=target_field)
    
    def test_balanced_numeric_sparse_bins(self) -> None:
        """Test balanced sampling when some bins might be empty."""
        target_field = Field('sparse_value', float64)
        sampler = BalancedNumericSampler(num_bins=10, target_field=target_field)
        
        # Only use values 0, 50, 100 (will create many empty bins)
        data_dict: dict[str, list] = {
            'id': list(range(30)),
            'sparse_value': ([0.0] * 10) + ([50.0] * 10) + ([100.0] * 10)
        }
        schema_fields = [Field('id', int32), Field('sparse_value', float64)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        result = sampler.sample(dataset, sample_size=15, random_seed=42)
        
        # Should still produce a valid sample despite sparse distribution
        assert result.data.height <= 15  # Might be less if some bins are empty
        values = set(result.data['sparse_value'].to_list())
        assert values.issubset({0.0, 50.0, 100.0})
    
    def test_balanced_numeric_non_numeric_field(self) -> None:
        """Test validation error for non-numeric target field."""
        target_field = Field('text', string)
        sampler = BalancedNumericSampler(num_bins=3, target_field=target_field)
        
        data_dict: dict[str, list] = {'id': [1, 2, 3], 'text': ['hello', 'world', 'test']}
        schema_fields = [Field('id', int32), Field('text', string)]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        with pytest.raises(ValueError, match="Field 'text' must be numeric"):
            sampler.sample(dataset, sample_size=2)


class TestSamplerIntegration:
    """Integration tests combining multiple sampling strategies."""
    
    def test_sampler_consistency(self) -> None:
        """Test that all samplers produce consistent results with same seed."""
        # Create a mixed dataset suitable for different samplers
        enum_dtype = EnumDtype('A', 'B', 'C')
        data_dict: dict[str, list] = {
            'id': list(range(100)),
            'category': (['A'] * 50) + (['B'] * 30) + (['C'] * 20),
            'score': [i + (i % 10) / 10.0 for i in range(100)]
        }
        schema_fields = [
            Field('id', int32),
            Field('category', enum_dtype),
            Field('score', float64)
        ]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        # Test reproducibility across different sampler types
        samplers = [
            RandomSampler(),
            ProportionalClassSampler(target_field=Field('category', enum_dtype)),
            BalancedClassSampler(target_field=Field('category', enum_dtype)),
            ProportionalNumericSampler(num_bins=4, target_field=Field('score', float64)),
            BalancedNumericSampler(num_bins=4, target_field=Field('score', float64)),
        ]
        
        for sampler in samplers:
            # Each sampler should produce consistent results with same seed
            result1 = sampler.sample(dataset, sample_size=20, random_seed=12345)
            result2 = sampler.sample(dataset, sample_size=20, random_seed=12345)
            
            assert result1.data.equals(result2.data), f'Sampler {type(sampler).__name__} not reproducible'
            assert result1.schema == result2.schema
            # Some balanced samplers might return slightly fewer samples due to balancing constraints
            assert result1.data.height <= 20 and result1.data.height > 0
    
    def test_extreme_sample_sizes(self) -> None:
        """Test samplers with edge case sample sizes."""
        enum_dtype = EnumDtype('X', 'Y')
        data_dict: dict[str, list] = {
            'id': list(range(50)),
            'category': (['X'] * 25) + (['Y'] * 25),
            'value': list(range(50))
        }
        schema_fields = [
            Field('id', int32),
            Field('category', enum_dtype),
            Field('value', int32)
        ]
        dataset = create_test_dataset(data_dict, schema_fields)
        
        # Test sample size = 1
        random_sampler = RandomSampler()
        result = random_sampler.sample(dataset, sample_size=1, random_seed=42)
        assert result.data.height == 1
        
        # Test sample size = dataset size
        result = random_sampler.sample(dataset, sample_size=50, random_seed=42)
        assert result.data.height == 50
        
        # Balanced sampler with odd sample size
        balanced_sampler = BalancedClassSampler(target_field=Field('category', enum_dtype))
        result = balanced_sampler.sample(dataset, sample_size=11, random_seed=42)
        # 11 // 2 = 5 per class, so we should get 10 total (5 + 5)
        assert result.data.height == 10
