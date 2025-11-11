"""Base classes for data sampling.

This module defines the core interfaces for data sampling operations.
"""

from abc import ABC, abstractmethod
from typing import override

import attrs

from agentune.analyze.core.dataset import Dataset


@attrs.define
class DataSampler(ABC):
    """Abstract base class for data sampling strategies."""
    @abstractmethod
    def sample(self, dataset: Dataset, sample_size: int, random_seed: int | None = None) -> Dataset:
        """Sample data from the given dataset."""

    def _validate_inputs(self, dataset: Dataset, sample_size: int) -> None:
        """Validate common sampling inputs."""
        if dataset.data.height == 0:
            raise ValueError('Cannot sample from empty dataset')

        if sample_size <= 0:
            raise ValueError('Sample size must be positive')


@attrs.define
class RandomSampler(DataSampler):
    """Simple random sampling without any stratification.
    
    This sampler selects rows uniformly at random from the dataset,
    ignoring any target variable or class distributions.
    """

    @override
    def sample(self, dataset: Dataset, sample_size: int, random_seed: int | None = None) -> Dataset:
        """Sample data using simple random sampling."""
        self._validate_inputs(dataset, sample_size)

        # If sample size >= dataset size, return the entire dataset
        if sample_size >= dataset.data.height:
            return dataset

        # Simple random sample
        sampled_df = dataset.data.sample(n=sample_size, seed=random_seed)

        return Dataset(
            schema=dataset.schema,
            data=sampled_df,
        )
