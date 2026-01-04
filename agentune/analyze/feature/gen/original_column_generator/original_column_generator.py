from collections.abc import Iterator
from typing import override

from attrs import define
from duckdb import DuckDBPyConnection

import agentune.core.types as t
from agentune.analyze.feature.gen.base import GeneratedFeature, SyncFeatureGenerator
from agentune.analyze.feature.gen.original_column_generator.features import (
    OriginalBoolFeature,
    OriginalFloatFeature,
    OriginalIntFeature,
)
from agentune.analyze.feature.problem import Problem
from agentune.analyze.join.base import TablesWithJoinStrategies
from agentune.core.dataset import Dataset
from agentune.core.schema import Field


@define
class OriginalColumnsGenerator(SyncFeatureGenerator):
    """A feature generator that exposes original dataset columns as features.
    
    This generator creates features from the raw columns in the dataset with minimal
    transformation. It focuses on faithful representation of the data, handling:
    - Numeric columns (integers and floats)
    - Boolean columns
    
    The generator automatically skips:
    - Target column (to prevent leakage)
    - Temporal columns (dates, times, timestamps)
    - Complex/nested types (lists, structs, arrays)
    - Constant columns (cardinality <= 1)
    """
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
            input_col=col.name,
            input_dtype=col.dtype
        )
    
    @staticmethod
    def _create_int_feature(col: Field) -> OriginalIntFeature:
        """Create an integer feature from a column."""
        return OriginalIntFeature(
            name=col.name,
            description=f'Original integer column {col.name}',
            technical_description=f'Direct pass-through of column {col.name}',
            default_for_missing=0,
            input_col=col.name,
            input_dtype=col.dtype
        )
    
    @staticmethod
    def _create_float_feature(col: Field) -> OriginalFloatFeature:
        """Create a float feature from a column."""
        return OriginalFloatFeature(
            name=col.name,
            description=f'Original float column {col.name}',
            technical_description=f'Direct pass-through of column {col.name}',
            default_for_missing=0.0,
            default_for_nan=0.0,
            default_for_infinity=0.0,
            default_for_neg_infinity=0.0,
            input_col=col.name,
            input_dtype=col.dtype
        )

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
            
            # Create feature based on type
            if col.dtype == t.boolean:
                yield GeneratedFeature(feature=OriginalColumnsGenerator._create_bool_feature(col), has_good_defaults=True)
            elif col.dtype.is_integer():
                yield GeneratedFeature(feature=OriginalColumnsGenerator._create_int_feature(col), has_good_defaults=True)
            elif col.dtype.is_float():
                yield GeneratedFeature(feature=OriginalColumnsGenerator._create_float_feature(col), has_good_defaults=True)
