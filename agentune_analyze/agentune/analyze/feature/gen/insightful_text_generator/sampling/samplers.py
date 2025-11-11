"""Base classes for data sampling.

This module defines the core interfaces for data sampling operations.
"""

from typing import override

import attrs
import polars as pl

from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Field
from agentune.analyze.feature.gen.insightful_text_generator.sampling.base import DataSampler

MIN_NUM_BINS = 2


def validate_field_exists_and_matches_schema(dataset: Dataset, field: Field) -> None:
    """Validate that field exists in dataset and matches schema dtype."""
    if field.name not in dataset.schema.names:
        raise ValueError(f"Target field '{field.name}' not found in dataset schema")
    schema_field = dataset.schema[field.name]
    if schema_field.dtype != field.dtype:
        raise ValueError(f'Target field dtype mismatch: expected {field.dtype}, got {schema_field.dtype}')


def validate_field_is_numeric(field: Field) -> None:
    """Validate that field is numeric."""
    if not field.dtype.is_numeric():
        raise ValueError(f"Field '{field.name}' must be numeric, got {field.dtype}")


@attrs.define
class ProportionalClassSampler(DataSampler):
    """Proportional sampling for categorical target variables.
    
    This sampler maintains the proportional distribution of categorical target values
    in the sampled data. It preserves class proportions from the original dataset.
    
    Requires target_field to be set in the class configuration.
    """
    target_field: Field  # Target field for sampling

    def sample(self, dataset: Dataset, sample_size: int, random_seed: int | None = None) -> Dataset:
        """Sample data using proportional sampling for categorical targets."""
        self._validate_inputs(dataset, sample_size)

        data = dataset.data
        total_rows = data.height
        if total_rows == 0 or total_rows <= sample_size:
            return dataset  # nothing to sample
        
        class_counts = data[self.target_field.name].value_counts().sort(self.target_field.name)
        
        sampled_parts = []
        
        for row in class_counts.iter_rows():
            class_value, count = row
            # Calculate proportional sample size for this class
            class_proportion = count / total_rows
            class_sample_size = max(1, round(sample_size * class_proportion))

            # Sample from this class
            class_data = data.filter(pl.col(self.target_field.name) == class_value)
            if class_data.height > 0:
                class_sample = class_data.sample(
                    n=min(class_sample_size, class_data.height),
                    seed=random_seed
                )
                sampled_parts.append(class_sample)
        
        sampled_df = pl.concat(sampled_parts)

        return Dataset(
            schema=dataset.schema,
            data=sampled_df,
        )

    @override
    def _validate_inputs(self, dataset: Dataset, sample_size: int) -> None:
        """Validate inputs for categorical stratified sampling."""
        super()._validate_inputs(dataset, sample_size)
        validate_field_exists_and_matches_schema(dataset, self.target_field)


@attrs.define
class ProportionalNumericSampler(DataSampler):
    """Proportional sampling for numeric target variables.

    This sampler uses quantile-based stratification to maintain the distribution
    of numeric target values across different ranges (strata).
    
    Requires target_field to be set in the class configuration.
    """
    num_bins: int  # Number of bins to create based on quantiles
    target_field: Field  # Target field for stratified sampling

    def __attrs_post_init__(self) -> None:
        """Validate num_bins after initialization."""
        if self.num_bins < MIN_NUM_BINS:
            raise ValueError(f'Number of bins must be at least {MIN_NUM_BINS}')

    @override
    def sample(self, dataset: Dataset, sample_size: int, random_seed: int | None = None) -> Dataset:
        """Sample data using stratified sampling for numeric targets."""
        self._validate_inputs(dataset, sample_size)

        data = dataset.data
        total_rows = data.height
        if total_rows == 0 or total_rows <= sample_size:
            return dataset  # nothing to sample
        
        target_series = data[self.target_field.name]
        
        # Sorted original row indices; nulls grouped last for stability
        idx_sorted = target_series.arg_sort()
        base, rem = divmod(sample_size, self.num_bins)
        selected_positions: list[int] = []

        for i in range(self.num_bins):
            # contiguous slice (by position) for bin i
            start = (i * total_rows) // self.num_bins
            end = ((i + 1) * total_rows) // self.num_bins

            num_to_sample = base + (1 if i < rem else 0)  # add one extra sample to first 'rem' bins
            # sample
            selected_positions.extend(pl.Series(range(start, end)).sample(n=num_to_sample, seed=random_seed))

        # Map sorted positions -> original row ids
        chosen_rows = [idx_sorted[p] for p in selected_positions]
        sampled_df = data.select(pl.all().gather(chosen_rows))
        return Dataset(
            schema=dataset.schema,
            data=sampled_df,
        )
    
    @override
    def _validate_inputs(self, dataset: Dataset, sample_size: int) -> None:
        """Validate inputs for numeric stratified sampling."""
        super()._validate_inputs(dataset, sample_size)
        validate_field_exists_and_matches_schema(dataset, self.target_field)
        validate_field_is_numeric(self.target_field)
        

@attrs.define
class BalancedClassSampler(DataSampler):
    """Balanced sampling for categorical target variables.
    
    This sampler creates equal representation across all categorical target values.
    Each class gets the same number of samples in the final dataset.
    
    Requires target_field to be set in the class configuration.
    """
    target_field: Field  # Target field for balanced sampling

    @override
    def sample(self, dataset: Dataset, sample_size: int, random_seed: int | None = None) -> Dataset:
        """Sample data using balanced sampling for categorical targets."""
        self._validate_inputs(dataset, sample_size)

        data = dataset.data
        total_rows = data.height
        if total_rows == 0 or total_rows <= sample_size:
            return dataset  # nothing to sample
        unique_classes = sorted(data[self.target_field.name].unique().to_list())
        num_classes = len(unique_classes)
        
        if num_classes == 0:
            raise ValueError('No classes found in target column')
        
        # Calculate samples per class
        samples_per_class = sample_size // num_classes

        sampled_parts = []
        
        for class_value in unique_classes:
            class_data = data.filter(pl.col(self.target_field.name) == class_value)
            
            if class_data.height > 0:
                class_sample_size = min(samples_per_class, class_data.height)
                class_sample = class_data.sample(n=class_sample_size, seed=random_seed)
                sampled_parts.append(class_sample)
        
        sampled_df = pl.concat(sampled_parts)

        return Dataset(
            schema=dataset.schema,
            data=sampled_df,
        )
    
    @override
    def _validate_inputs(self, dataset: Dataset, sample_size: int) -> None:
        """Validate inputs for categorical balanced sampling."""
        super()._validate_inputs(dataset, sample_size)
        validate_field_exists_and_matches_schema(dataset, self.target_field)


@attrs.define
class BalancedNumericSampler(DataSampler):
    """Balanced sampling for numeric target variables.
    
    This sampler creates equal representation across numeric value ranges (bins).
    Each bin gets the same number of samples in the final dataset.
    
    Requires target_field to be set in the class configuration.
    """
    num_bins: int  # Number of equal-width bins to create
    target_field: Field  # Target field for balanced sampling

    def __attrs_post_init__(self) -> None:
        """Validate num_bins after initialization."""
        if self.num_bins < MIN_NUM_BINS:
            raise ValueError(f'Number of bins must be at least {MIN_NUM_BINS}')

    @override
    def sample(self, dataset: Dataset, sample_size: int, random_seed: int | None = None) -> Dataset:
        """Sample data using balanced sampling for numeric targets."""
        self._validate_inputs(dataset, sample_size)

        data = dataset.data
        total_rows = data.height
        if total_rows == 0 or total_rows <= sample_size:
            return dataset  # nothing to sample
        target_series = data[self.target_field.name]
        
        # Get min and max values
        min_val = float(target_series.min())  # type: ignore[arg-type]
        max_val = float(target_series.max())  # type: ignore[arg-type]
        
        if min_val == max_val:
            # All values are the same, just random sample
            sampled_df = data.sample(n=sample_size, seed=random_seed)
        else:
            # Create equal-width bins
            bin_width = (max_val - min_val) / self.num_bins
            samples_per_bin = sample_size // self.num_bins

            sampled_parts = []

            bounds = [[min_val + i * bin_width, min_val + (i + 1) * bin_width] for i in range(self.num_bins)]
            bounds[0][0] = float('-inf')  # Ensure first bin includes min_val
            bounds[-1][1] = float('inf')  # Ensure last bin includes max_val

            for lower_bound, upper_bound in bounds:
                condition = (pl.col(self.target_field.name) > lower_bound) & (pl.col(self.target_field.name) <= upper_bound)
                bin_data = data.filter(condition)
                
                if bin_data.height > 0:
                    bin_sample_size = min(samples_per_bin, bin_data.height)
                    bin_sample = bin_data.sample(n=bin_sample_size, seed=random_seed)
                    sampled_parts.append(bin_sample)
            
            sampled_df = pl.concat(sampled_parts)

        return Dataset(
            schema=dataset.schema,
            data=sampled_df,
        )

    @override
    def _validate_inputs(self, dataset: Dataset, sample_size: int) -> None:
        """Validate inputs for numeric balanced sampling."""
        super()._validate_inputs(dataset, sample_size)
        validate_field_exists_and_matches_schema(dataset, self.target_field)
        validate_field_is_numeric(self.target_field)
