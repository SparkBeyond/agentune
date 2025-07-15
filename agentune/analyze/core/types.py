from __future__ import annotations

from collections.abc import Sequence

import duckdb
import duckdb.typing as ddt
import polars as pl
import polars.datatypes
import pyarrow as pa
from attrs import field, frozen

from agentune.analyze.core import setup

# We define these types instad of using pl.Field and pl.Schema because we might want to support e.g. semantic types in the future.

# Used in some Polars APIs. Copy of a type union defined in polars._types.
type PolarsDataType = pl.DataType | polars.datatypes.DataTypeClass

@frozen
class Dtype:
    name: str
    duckdb_type: ddt.DuckDBPyType = field()
    polars_type: PolarsDataType = field()
    python_type: type = field(init=False)

    @python_type.default
    def _python_type_default(self) -> type:
        # TODO check if this is correct when working directly with duckdb - there might not be a single answer for both!
        return self.polars_type.to_python()
    
    def arrow_type(self) -> pa.DataType:
        # Clunky, but it works without hardcoding the coversions (though we might want to cache them?)
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

boolean = Dtype('bool', ddt.BOOLEAN, pl.Boolean)

int8 = Dtype('int8', ddt.TINYINT, pl.Int8)
int16 = Dtype('int16', ddt.SMALLINT, pl.Int16)
int32 = Dtype('int32', ddt.INTEGER, pl.Int32)
int64 = Dtype('int64', ddt.BIGINT, pl.Int64)

uint8 = Dtype('uint8', ddt.UTINYINT, pl.UInt8)
uint16 = Dtype('uint16', ddt.USMALLINT, pl.UInt16)
uint32 = Dtype('uint32', ddt.UINTEGER, pl.UInt32)
uint64 = Dtype('uint64', ddt.UBIGINT, pl.UInt64)

float32 = Dtype('float32', ddt.FLOAT, pl.Float32)
float64 = Dtype('float64', ddt.DOUBLE, pl.Float64)

string = Dtype('str', ddt.VARCHAR, pl.String)
json = Dtype('json', duckdb.dtype('JSON'), pl.String)

uuid = Dtype('uuid', ddt.UUID, pl.String)

# Call setup to install the duckdb 'spatial' extension, otherwise the call to duckdb.dtype() will fail.
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
# st_polars requires using GEOMETRY and has a heavy-ish additional dependency, so I'm skipping it for now.
# Geopolars is in alpha, and is blocked on Polars adding support for Arrow extension types (which will unblock many things).
point = Dtype('point', duckdb.dtype('POINT_2D'), pl.Struct([pl.Field('x', pl.Float64), pl.Field('y', pl.Float64)]))

date = Dtype('date', ddt.DATE, pl.Date)
time = Dtype('time', ddt.TIME, pl.Time)

# TODO confirm standardization on ms precision
timestamp = Dtype('timestamp', ddt.TIMESTAMP_MS, pl.Datetime('ms'))

@frozen(init=False)
class EnumDtype(Dtype):
    values: tuple[str, ...]

    @staticmethod
    def duckdb_enum_type(values: Sequence[str]) -> ddt.DuckDBPyType:
        # Can't create it directly by calling duckdb.enum_type(), we get a NotImplementedException ("enum_type creation method is not implemented yet")
        # This isn't risking an actual SQL escape but if we don't format this properly, the type definition won't parse; TODO test
        escaped = ', '.join("'" + value.replace("'", "''") + "'" for value in values)
        return duckdb.dtype(f'ENUM({escaped})')
    
    def __init__(self, *values: str):
        self.__attrs_init__(
            f'Enum[{', '.join(values)}]',
            EnumDtype.duckdb_enum_type(list(values)),
            pl.Enum(categories=values),
            values
        )

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

_simple_dtypes = [boolean, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, string, json, uuid, date, time, timestamp, point]
_simple_dtype_by_polars_type = {dtype.polars_type: dtype for dtype in _simple_dtypes
                                if dtype is not json and dtype is not uuid} # those are erased to string
_simple_dtype_by_duckdb_type_str = {str(dtype.duckdb_type): dtype for dtype in _simple_dtypes}


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
        return EnumDtype(* dict(ddtype.children)['values'])
    elif ddtype.id == 'list':
        return ListDtype(dtype_from_duckdb(dict(ddtype.children)['child']))
    elif ddtype.id == 'array':
        params = dict(ddtype.children)
        return ArrayDtype(dtype_from_duckdb(params['child']), params['size'])
    elif ddtype.id == 'struct':
        return StructDtype(* [(name, dtype_from_duckdb(dtype)) for name, dtype in ddtype.children])
    else:
        raise ValueError(f'Unsupported duckdb type: {ddtype}')

