import logging
from collections.abc import Iterator
from typing import override

import polars as pl
from attrs import define
from duckdb import DuckDBPyConnection

import agentune.core.types as t
from agentune.analyze.feature.base import CategoricalFeature
from agentune.analyze.feature.gen.base import GeneratedFeature, SyncFeatureGenerator
from agentune.analyze.feature.gen.original_column_generator.features import (
    OriginalBoolFeature,
    OriginalCategoricalFeature,
    OriginalFloatFeature,
    OriginalIntFeature,
)
from agentune.analyze.feature.problem import Problem
from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.core.dataset import Dataset
from agentune.core.schema import Field

_logger = logging.getLogger(__name__)


@define
class OriginalColumnsGenerator(SyncFeatureGenerator):
    """A feature generator that exposes original primary dataset columns as features.

    This generator creates features from the raw columns in the dataset with minimal
    transformation. It focuses on faithful representation of the data, handling:
    - Numeric columns (integers and floats)
    - Boolean columns
    - Enum columns (using all schema-defined categories)
    - String columns (top-K most frequent + CategoricalFeature.other_category for others)

    The generator automatically skips:
    - Target column (to prevent leakage)
    - Temporal columns (dates, times, timestamps)
    - Complex/nested types (lists, structs, arrays)
    - Constant columns (cardinality <= 1)
    - String columns where top-K values don't provide sufficient coverage
    
    Args:
        top_k_categories: Maximum number of categories to extract from string columns
        min_coverage_threshold: Minimum fraction of data that top-K categories must cover
    """
    top_k_categories: int = 9
    min_coverage_threshold: float = 0.5

    @staticmethod
    def _should_skip_column(col: Field, dataset: Dataset, target_column_name: str) -> bool:
        """Determine if a column should be skipped.
        
        Returns True if the column is the target, temporal, nested, or constant.
        """
        # Skip target column
        if col.name == target_column_name:
            return True

        # Skip temporal and complex types
        if col.dtype.is_temporal() or col.dtype.is_nested():
            return True
        
        # Skip constant columns
        if dataset.data.get_column(col.name).n_unique() <= 1:
            return True
        
        return False
    
    @staticmethod
    def _create_bool_feature(col: Field) -> OriginalBoolFeature:
        """Create a boolean feature from a column."""
        return OriginalBoolFeature(
            name=col.name,
            description=f'Original boolean column {col.name}',
            technical_description=f'Direct pass-through of column {col.name}',
            default_for_missing=False,
            input=col
        )
    
    @staticmethod
    def _create_int_feature(col: Field) -> OriginalIntFeature:
        """Create an integer feature from a column."""
        return OriginalIntFeature(
            name=col.name,
            description=f'Original integer column {col.name}',
            technical_description=f'Direct pass-through of column {col.name}',
            default_for_missing=0,
            input=col
        )
    
    @staticmethod
    def _create_float_feature(col: Field) -> OriginalFloatFeature:
        """Create a float feature from a column."""
        return OriginalFloatFeature(
            name=col.name,
            description=f'Original float column {col.name}',
            technical_description=f'Direct pass-through of column {col.name}',
            input=col
        )

    @staticmethod
    def _create_categorical_feature(col: Field, categories: tuple[str, ...]) -> OriginalCategoricalFeature:
        """Create a categorical feature from a column."""
        source_type = 'enum' if col.dtype.is_enum() else 'string'
        
        if source_type == 'string':
            description = f'Top {len(categories)} categories of string column {col.name}'
            technical_description = f'Top {len(categories)} categories from string column {col.name}, mapping all other values to "{CategoricalFeature.other_category}"'
        else:
            description = f'Original categorical column {col.name} (from {source_type})'
            technical_description = f'Direct pass-through of {source_type} column {col.name}'

        return OriginalCategoricalFeature(
            name=col.name,
            description=description,
            technical_description=technical_description,
            default_for_missing=CategoricalFeature.other_category,
            input=col,
            categories=categories
        )

    def _analyze_string_column(self, series: pl.Series) -> tuple[str, ...] | None:
        """Analyze a string column and return top-K categories if they provide sufficient coverage.

        Returns None if top-K categories don't provide sufficient coverage.
        """
        value_counts = series.value_counts(sort=True)
        top_k = value_counts.head(self.top_k_categories)

        # Check if top-K values cover sufficient portion of data
        coverage = top_k.get_column('count').sum() / len(series)
        if coverage < self.min_coverage_threshold:
            return None

        return tuple(top_k.get_column(series.name).to_list())

    @override
    def generate(
        self,
        feature_search: Dataset,
        problem: Problem,
        join_strategies: TablesWithJoinStrategies,
        conn: DuckDBPyConnection,
    ) -> Iterator[GeneratedFeature]:
        for col in feature_search.schema.cols:
            # Skip columns that shouldn't be processed
            if OriginalColumnsGenerator._should_skip_column(col, feature_search, problem.target_column.name):
                continue

            # Determine feature creation logic based on column type
            match col.dtype:
                case t.boolean:
                    yield GeneratedFeature(
                        feature=OriginalColumnsGenerator._create_bool_feature(col),
                        has_good_defaults=False
                    )
                case dt if dt.is_integer():
                    yield GeneratedFeature(
                        feature=OriginalColumnsGenerator._create_int_feature(col),
                        has_good_defaults=False
                    )
                case dt if dt.is_float():
                    yield GeneratedFeature(
                        feature=OriginalColumnsGenerator._create_float_feature(col),
                        has_good_defaults=False
                    )
                case t.EnumDtype(values=categories):
                    # Use all categories from the enum schema definition
                    yield GeneratedFeature(
                        feature=OriginalColumnsGenerator._create_categorical_feature(col, categories),
                        has_good_defaults=False
                    )
                case t.string:
                    # Analyze string column for top-K categorical conversion
                    series = feature_search.data.get_column(col.name)
                    top_categories = self._analyze_string_column(series)
                    if top_categories is not None:
                        yield GeneratedFeature(
                            feature=OriginalColumnsGenerator._create_categorical_feature(col, top_categories),
                            has_good_defaults=False
                        )
                    # If top_categories is None, skip the column (insufficient coverage)
                case _:
                    # Skip unexpected types
                    _logger.debug(
                        f"Skipping column '{col.name}' with unsupported type {col.dtype}"
                    )
                    continue
