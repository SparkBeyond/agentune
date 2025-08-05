from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import attrs
import polars as pl
import pyarrow as pa
from attrs import field, frozen
from duckdb import DuckDBPyRelation

from agentune.analyze.core.types import Dtype, EnumDtype

# We define these types instad of using pl.Field and pl.Schema because we might want to support e.g. semantic types in the future.

@frozen
class Field:
    """We treat all fields as always nullable. (Polars support for non-nullable fields is imperfect.)"""
    
    name: str
    dtype: Dtype
    
    def to_polars(self) -> pl.Field:
        return pl.Field(self.name, self.dtype.polars_type)
    
    @staticmethod
    def from_polars(field: pl.Field) -> Field:
        return Field(field.name, Dtype.from_polars(field.dtype))

@frozen
class Schema:
    cols: tuple[Field, ...]
    _by_name: Mapping[str, Field] = field(init=False, eq=False, hash=False, repr=False)

    @_by_name.default
    def _by_name_default(self) -> dict[str, Field]:
        # duckdb allows duplicate names of columns in a relation, but polars doesn't allow duplicate column names in a DataFrame
        assert len({col.name for col in self.cols}) == len(tuple(self.cols)), 'Duplicate column names'
        return {col.name: col for col in self.cols}

    @property
    def names(self) -> list[str]: 
        return [col.name for col in self.cols]

    @property
    def dtypes(self) -> list[Dtype]: 
        return [col.dtype for col in self.cols]
    
    def drop(self, *names: str) -> Schema:
        return Schema(tuple(col for col in self.cols if col.name not in names))

    def select(self, *cols: str) -> Schema:
        return Schema(tuple(col for col in self.cols if col.name in cols))

    def hstack(self, other: Schema) -> Schema:
        common_names = set(self.names).intersection(other.names)
        if common_names:
            raise ValueError(f'Cannot hstack, duplicate column names: {common_names}, {self.names=}, {other.names=}')
        return Schema(self.cols + other.cols)

    def __len__(self) -> int: 
        return len(tuple(self.cols))

    def __getitem__(self, col_name: str) -> Field:
        return self._by_name[col_name]

    def to_polars(self) -> pl.Schema:
        return pl.Schema((col.name, col.dtype.polars_type) for col in self.cols)
    
    def to_arrow(self) -> pa.Schema:
        return pa.schema(pa.field(col.name, col.dtype.arrow_type()) for col in self.cols)
        
    @staticmethod
    def from_duckdb(relation: DuckDBPyRelation) -> Schema: 
        return Schema(tuple(Field(col, Dtype.from_duckdb(ddtype)) for col, ddtype in zip(relation.columns, relation.types, strict=True)))

    @staticmethod
    def from_polars(input: pl.DataFrame | pl.LazyFrame | pl.Schema) -> Schema: 
        """Note that some schema information is not represented in a polars DataFrame or LazyFrame.
        A schema created from them will have some erased types.
        """
        pl_schema = input if isinstance(input, pl.Schema) else input.schema

        return Schema(tuple(Field(col, Dtype.from_polars(dtype)) for col, dtype in pl_schema.items()))


def restore_df_types(df: pl.DataFrame, schema: Schema) -> pl.DataFrame:
    """Restore the correct types to a Polars dataframe created from a DuckDB relation, given the schema."""
    # Preserve enum types
    for col in schema.cols:
        if isinstance(col.dtype, EnumDtype):
            values = col.dtype.values
            df = df.cast({col.name: pl.Enum(categories=values)})
    return df


def dtype_is(dtype: Dtype) -> Callable[[Any, attrs.Attribute, Field], None]:
    """An attrs field validator that checks that a Field has the given dtype."""
    def validator(_self: Any, _attribute: attrs.Attribute, value: Field) -> None:
        if value.dtype != dtype:
            raise ValueError(f'Column {value.name} has dtype {value.dtype}, but should have dtype {dtype}')
    return validator
