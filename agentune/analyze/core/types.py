from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import partial
from typing import Any, cast, override

import cattrs.strategies
import duckdb
import duckdb.sqltypes as ddt
import polars as pl
import polars.datatypes
import pyarrow as pa
from attrs import field, frozen

from agentune.analyze.core import sercontext, setup

# Used in some Polars APIs. Copy of a type union defined in polars._types.
type PolarsDataType = pl.DataType | polars.datatypes.DataTypeClass

@frozen
class Dtype(ABC):
    name: str
    duckdb_type: ddt.DuckDBPyType = field(eq=False, hash=False)
    polars_type: PolarsDataType = field(eq=False, hash=False)

    def __str__(self) -> str:
        return self.name
    
    def arrow_type(self) -> pa.DataType:
        # Clunky, but it works without hardcoding the conversions (though we might want to cache them?)
        return pl.Series('a', [], self.polars_type).to_arrow().type
    
    @staticmethod
    def from_polars(pltype: PolarsDataType) -> Dtype:
        return dtype_from_polars(pltype)
    
    @staticmethod
    def from_duckdb(ddtype: ddt.DuckDBPyType) -> Dtype:
        return dtype_from_duckdb(ddtype)

    def is_numeric(self) -> bool:
        return self.polars_type.is_numeric()
    
    def is_integer(self) -> bool:
        return self.polars_type.is_integer()

    def is_signed_integer(self) -> bool:
        return self.polars_type.is_signed_integer()
    
    def is_unsigned_integer(self) -> bool:
        return self.polars_type.is_unsigned_integer()

    def is_float(self) -> bool:
        return self.polars_type.is_float()

    def is_temporal(self) -> bool:
        return self.polars_type.is_temporal()

    def is_nested(self) -> bool:
        """Is a list, array, map, or struct."""
        return self.polars_type.is_nested()

    def is_enum(self) -> bool:
        return False

    @abstractmethod
    def _to_ser_dtype(self) -> _SerDtype: ...

@frozen
class SimpleDtype(Dtype):
    """A Dtype that can be reconstructed from its string name, listed in _simple_dtypes."""

    @override
    def _to_ser_dtype(self) -> _SerDtype:
        return _SimpleSerDtype(self.name)


boolean = SimpleDtype('bool', ddt.BOOLEAN, pl.Boolean)

int8 = SimpleDtype('int8', ddt.TINYINT, pl.Int8)
int16 = SimpleDtype('int16', ddt.SMALLINT, pl.Int16)
int32 = SimpleDtype('int32', ddt.INTEGER, pl.Int32)
int64 = SimpleDtype('int64', ddt.BIGINT, pl.Int64)

uint8 = SimpleDtype('uint8', ddt.UTINYINT, pl.UInt8)
uint16 = SimpleDtype('uint16', ddt.USMALLINT, pl.UInt16)
uint32 = SimpleDtype('uint32', ddt.UINTEGER, pl.UInt32)
uint64 = SimpleDtype('uint64', ddt.UBIGINT, pl.UInt64)

float32 = SimpleDtype('float32', ddt.FLOAT, pl.Float32)
float64 = SimpleDtype('float64', ddt.DOUBLE, pl.Float64)

string = SimpleDtype('str', ddt.VARCHAR, pl.String)
json_dtype = SimpleDtype('json', duckdb.dtype('JSON'), pl.String)
uuid_dtype = SimpleDtype('uuid', ddt.UUID, pl.String)

# Call setup to install the duckdb 'spatial' extension, otherwise the call to duckdb.SimpleDtype() will fail.
setup.setup()

# We use the point_2d type, not the general purpose GEOMETRY type, so we can identify such columns, and so that we can
# enforce the column ought to only store points. 
# This requires a slightly inefficient cast in most geospatial functions (and presumably for the Rtree index),
# see https://duckdb.org/docs/stable/extensions/spatial/overview.html. If it comes time to optimize, we might switch to GEOMETRY.
#
# On the polars side, this lets us get data of a struct type we can recognize and parse, whereas GEOMETRY would create 
# a column of dtype binary (EWKB encoded) and force us to use an extra library to parse it.
# (This means dtype_from_polars will think that any polars column whose type is this kind of struct is a point, but that's better than
#  thinking that any binary column in a point!)
#
# In future, to let us use the struct type productively on the Python side, we should register a Polars extension that
# emits a custom scalar type Point (e.g. a namedtuple), and that includes basic functions like distance.
# 
# There are two prospective Polars geospatial extensions: st_polars and geopolars.
# st_polars require using GEOMETRY and has a heavy-ish additional dependency, so I'm skipping it for now.
# Geopolars is in alpha, and is blocked on Polars adding support for Arrow extension types (which will unblock many things).
# Point types are currently disabled / unsupported
# point = SimpleDtype('point', duckdb.SimpleDtype('POINT_2D'), pl.Struct([pl.Field('x', pl.Float64), pl.Field('y', pl.Float64)]))

date_dtype = SimpleDtype('date', ddt.DATE, pl.Date)
time_dtype = SimpleDtype('time', ddt.TIME, pl.Time)

timestamp = SimpleDtype('timestamp', ddt.TIMESTAMP_MS, pl.Datetime('ms'))

@frozen(init=False)
class EnumDtype(Dtype):
    values: tuple[str, ...]

    @staticmethod
    def duckdb_enum_type(values: Sequence[str]) -> ddt.DuckDBPyType:
        # Can't create it directly by calling duckdb.enum_type(), we get a NotImplementedException ("enum_type creation method is not implemented yet")
        # This isn't risking an actual SQL escape but if we don't format this properly, the type definition won't parse.
        escaped = ', '.join("'" + value.replace("'", "''") + "'" for value in values)
        return duckdb.dtype(f'ENUM({escaped})')

    def __init__(self, *values: str):
        self.__attrs_init__(
            f'Enum[{', '.join(values)}]',
            EnumDtype.duckdb_enum_type(list(values)),
            pl.Enum(categories=values),
            values
        )

    @override
    def is_enum(self) -> bool:
        return True

    @override
    def _to_ser_dtype(self) -> _SerDtype:
        return _EnumSerDtype(self.values)


@frozen(init=False)
class ListDtype(Dtype):
    inner: Dtype

    def __init__(self, inner: Dtype):
        self.__attrs_init__(
            f'List[{inner.name}]', 
            duckdb.list_type(inner.duckdb_type), 
            pl.List(inner.polars_type),
            inner
        )

    @override
    def _to_ser_dtype(self) -> _SerDtype:
        return _ListSerDtype(self.inner._to_ser_dtype())


@frozen(init=False)
class ArrayDtype(Dtype):
    inner: Dtype
    size: int

    def __init__(self, inner: Dtype, size: int):
        self.__attrs_init__(
            f'Array[{inner.name}, {size}]', 
            duckdb.array_type(inner.duckdb_type, size), 
            pl.Array(inner.polars_type, size),
            inner,
            size
        )

    @override
    def _to_ser_dtype(self) -> _SerDtype:
        return _ArraySerDtype(self.inner._to_ser_dtype(), self.size)

@frozen(init=False)
class StructDtype(Dtype):
    fields: tuple[tuple[str, Dtype], ...]

    def __init__(self, *fields: tuple[str, Dtype]):
        self.__attrs_init__(
            f'Struct[{', '.join(f'{name}: {dtype}' for name, dtype in fields)}]', 
            duckdb.struct_type({name: dtype.duckdb_type for name, dtype in fields}), 
            pl.Struct([pl.Field(name, dtype.polars_type) for name, dtype in fields]),
            fields
        )

    @override
    def _to_ser_dtype(self) -> _SerDtype:
        return _StructSerDtype(tuple((name, dt._to_ser_dtype()) for name, dt in self.fields))

_simple_dtypes = [boolean, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, string, json_dtype, uuid_dtype, date_dtype, time_dtype, timestamp] # , point]
_simple_dtype_by_polars_type = {dtype.polars_type: dtype for dtype in _simple_dtypes
                                if dtype is not json_dtype and dtype is not uuid_dtype} # those are erased to string
_simple_dtype_by_duckdb_type_str = {str(dtype.duckdb_type): dtype for dtype in _simple_dtypes}
_simple_dtype_by_duckdb_type_str[str(ddt.TIMESTAMP)] = timestamp # alias for TIMESTAMP_MS
_simple_dtype_by_name = {dtype.name: dtype for dtype in _simple_dtypes}

def dtype_from_polars(pltype: PolarsDataType) -> Dtype:
    if pltype in _simple_dtype_by_polars_type:
        return _simple_dtype_by_polars_type[pltype]
    elif isinstance(pltype, pl.Enum):
        return EnumDtype(*pltype.categories.to_list())
    elif isinstance(pltype, pl.Categorical):
        return string # if an enum is erased into a Categorical, we can't recover the enum values
    elif isinstance(pltype, pl.List):
        return ListDtype(dtype_from_polars(pltype.inner))
    elif isinstance(pltype, pl.Array):
        return ArrayDtype(dtype_from_polars(pltype.inner), pltype.size)
    elif isinstance(pltype, pl.Struct):
        return StructDtype(* [(field.name, dtype_from_polars(field.dtype)) for field in pltype.fields])
    else:
        raise ValueError(f'Unsupported polars type: {pltype}')

def dtype_from_duckdb(ddtype: ddt.DuckDBPyType) -> Dtype:
    if str(ddtype) in _simple_dtype_by_duckdb_type_str:
        return _simple_dtype_by_duckdb_type_str[str(ddtype)]
    elif ddtype.id == 'enum':
        enum_children = cast(Sequence[tuple[str, str]], ddtype.children)
        return EnumDtype(* dict(enum_children)['values'])
    elif ddtype.id == 'list':
        children = cast(Sequence[tuple[str, ddt.DuckDBPyType]], ddtype.children)
        return ListDtype(dtype_from_duckdb(dict(children)['child']))
    elif ddtype.id == 'array':
        params = dict(ddtype.children)
        return ArrayDtype(dtype_from_duckdb(cast(ddt.DuckDBPyType, params['child'])), cast(int, params['size']))
    elif ddtype.id == 'struct':
        children = cast(Sequence[tuple[str, ddt.DuckDBPyType]], ddtype.children)
        return StructDtype(* [(name, dtype_from_duckdb(dtype)) for name, dtype in children])
    else:
        raise ValueError(f'Unsupported duckdb type: {ddtype}')

def python_type_from_polars(dtype: Dtype) -> type:
    """The Python type of scalar values returned from Polars series of this dtype."""
    return dtype.polars_type.to_python()

def python_type_from_duckdb(dtype: Dtype) -> type:
    """The Python type of scalar values returned from DuckDB query results of this dtype."""
    if isinstance(dtype, ArrayDtype):
        return tuple # Unlike the Polars type, which is list
    if dtype == uuid_dtype:
        return uuid.UUID # Unlike the Polars type, which is erased to str
    return dtype.polars_type.to_python()

# We need these because Dtype cannot be automatically serialized; Polars and duckdb dtypes are not serializable
# but also, we don't need to serialize them, we just need to tell cattrs that simple dtypes can be looked up by name.

@frozen
class _SerDtype(ABC):
    """Automatically serializable equivalent of Dtype."""

    @abstractmethod
    def to_dtype(self) -> Dtype: ...

@frozen
class _SimpleSerDtype(_SerDtype):
    name: str

    def to_dtype(self) -> Dtype:
        return _simple_dtype_by_name[self.name]


@frozen
class _EnumSerDtype(_SerDtype):
    values: tuple[str, ...]

    def to_dtype(self) -> Dtype:
        return EnumDtype(*self.values)

@frozen
class _ListSerDtype(_SerDtype):
    inner: _SerDtype

    def to_dtype(self) -> Dtype:
        return ListDtype(self.inner.to_dtype())

@frozen
class _ArraySerDtype(_SerDtype):
    inner: _SerDtype
    size: int

    def to_dtype(self) -> Dtype:
        return ArrayDtype(self.inner.to_dtype(), self.size)

@frozen
class _StructSerDtype(_SerDtype):
    fields: tuple[tuple[str, _SerDtype], ...]

    def to_dtype(self) -> Dtype:
        return StructDtype(*tuple((name, dt.to_dtype()) for name, dt in self.fields))

cattrs.strategies.include_subclasses(_SerDtype, sercontext.converter_for_hook_registration,
                                     union_strategy=partial(cattrs.strategies.configure_tagged_union,
                                                            tag_generator=lambda cls: cls.__name__.removeprefix('_').removesuffix('SerDtype').lower()))

@sercontext.converter_for_hook_registration.register_unstructure_hook
def _unstructure_dtype(dtype: Dtype) -> Any:
    return sercontext.converter_for_hook_registration.unstructure(dtype._to_ser_dtype())

@sercontext.converter_for_hook_registration.register_structure_hook
def _structure_dtype(d: dict[str, Any], _: type[Dtype]) -> Dtype:
    return sercontext.converter_for_hook_registration.structure(d, _SerDtype).to_dtype() # type: ignore[type-abstract]

