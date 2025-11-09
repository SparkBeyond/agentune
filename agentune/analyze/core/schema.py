from __future__ import annotations

from collections.abc import Callable
from typing import Any

import attrs
import polars as pl
import pyarrow as pa
from attrs import field, frozen
from duckdb import DuckDBPyRelation
from frozendict import frozendict

from agentune.analyze.core.types import ArrayDtype, Dtype, EnumDtype, ListDtype, StructDtype
from agentune.analyze.util.attrutil import frozendict_converter

# We define these types instead of using pl.Field and pl.Schema because support a slightly different set of types,
# and in the future we might want to support logical types that polars doesn't.

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
    _by_name: frozendict[str, Field] = field(init=False, eq=False, hash=False, repr=False, converter=frozendict_converter)

    @_by_name.default
    def _by_name_default(self) -> dict[str, Field]:
        # duckdb allows duplicate names of columns in a relation, but polars doesn't allow duplicate column names in a DataFrame
        if len({col.name for col in self.cols}) != len(tuple(self.cols)):
            raise ValueError(f'Duplicate column names: {self.names}')
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

    def __add__(self, other: Schema | Field) -> Schema:
        if isinstance(other, Schema):
            return Schema(self.cols + other.cols)
        else:
            return Schema((*self.cols, other))

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


def _contains_enum_type(dtype: Dtype) -> bool:
    match dtype:
        case EnumDtype(): return True
        case ListDtype(inner=inner): return _contains_enum_type(inner)
        case ArrayDtype(inner=inner): return _contains_enum_type(inner)
        case StructDtype(fields=fields): return any(_contains_enum_type(inner) for _, inner in fields)
        case other if other.is_nested(): raise ValueError(f'Unexpected nested type: {other}')
        case _: return False

def restore_df_types(df: pl.DataFrame, schema: Schema) -> pl.DataFrame:
    """Restore the correct types to a Polars dataframe created from a DuckDB relation, given the original schema."""
    # Preserve enum types
    for col in schema.cols:
        if _contains_enum_type(col.dtype):
            df = df.cast({col.name: col.dtype.polars_type})
    return df

def restore_relation_types(relation: DuckDBPyRelation, schema: Schema) -> DuckDBPyRelation:
    """Given a relation that matches an 'erased' version of this schema, return a new relation
    that casts its columns to the types specified in the schema.
    """
    if not any(_contains_enum_type(dtype) for dtype in schema.dtypes):
        return relation

    expr = ', '.join(f'"{field.name}"::{field.dtype.duckdb_type} as "{field.name}"' for field in schema.cols)
    return relation.project(expr)


def dtype_is(dtype: Dtype | type[Dtype]) -> Callable[[Any, attrs.Attribute, Field], None]:
    """An attrs field validator that checks that a Field has the given dtype."""
    def validator(_self: Any, _attribute: attrs.Attribute, value: Field) -> None:
        if isinstance(dtype, type):
            condition = isinstance(value.dtype, dtype)
        else:
            condition = value.dtype == dtype
        if not condition:
            raise ValueError(f'Column {value.name} has dtype {value.dtype}, but should have dtype {dtype}')
    return validator
