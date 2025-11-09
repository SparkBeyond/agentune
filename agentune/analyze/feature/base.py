import asyncio
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, final, override

import attrs
import polars as pl
from attrs import define
from duckdb import DuckDBPyConnection

import agentune.analyze.core.types
from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Schema
from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.core.types import Dtype
from agentune.analyze.feature.compute.limits import amap_gather_with_limit
from agentune.analyze.join.base import JoinStrategy
from agentune.analyze.util.cattrutil import UseTypeTag


# Feature ABCs set slots=False to allow diamond inheritance from e.g. IntFeature + LlmFeature;
# Python forbids multiple inheritance from slots classes.
# To be able to override abstract properties with attrs attributes, you need to make your final
# class a slots class (i.e. @frozen without slots=False).
@define(slots=False)
class Feature[T](ABC, UseTypeTag):
    """A feature calculates a value that can be used to predict the target in a dataset.

    Handling errors, missing values, and non-finite float values in feature outputs:
        The methods (a)compute, (a)compute_batch can raise an error, return a missing value (None for `compute`),
        return a NaN or +/- infinity value for float features, and return an unexpected string for categorical features.

        The _safe variants return None instead of raising an error.
        For categorical features, they also return the special value `CategoricalFeature.other_category`
        if `compute` returns an unexpected value (one not in the feature's categories list).

        The _with_defaults variants substitute the default_for_xxx attributes in these five cases.

    Implementation note:
        This base class is annotated with @attrs.define, and all implementations must be attrs classes.
        We rely on attrs.evolve() being able to change e.g. feature names and descriptions.
        Only attributes that must be free parameters to the feature are explicitly declared as attributes.

    Args:
        name: Used as the column/series name in outputs. Not guaranteed to be unique among Feature instances.
        description: Human-readable description of the feature.
        technical_description: Human-readable description of feature's implementation details.
        default_for_missing: a value substituted by compute_with_defaults if the underlying `compute` outputs
                             a missing value.

    Type parameters:
        T: The type of the feature's output values, when they appear as scalars.
           This is not a free type parameter; only the values defined by the subclasses below, such as IntFeature, are allowed.
           Note that features with different dtypes can have the same scalar T, e.g. features with dtype Int32 and Int64 would
           both have T=int. (There is no feature type using Int64 at the moment of writing, but you should not write code
           that assumes all features have distinct T types.)
    """

    name: str
    description: str
    technical_description: str

    default_for_missing: T

    @property
    @abstractmethod
    def dtype(self) -> Dtype:
        """The dtype of series returned by acompute_batch_safe. See also raw_dtype."""
        ...

    @property
    def raw_dtype(self) -> Dtype:
        """The dtype of series returned by acompute_batch (the non-safe version).

        Can be more general than `self.dtype`, with the _safe computation coercing raw values to the right dtype.
        """
        return self.dtype

    @property
    @abstractmethod
    def params(self) -> Schema: 
        """Columns of the main table used by the feature.
        This affects the parameters to compute().
        """
        ...
    
    @property
    @abstractmethod
    def secondary_tables(self) -> Sequence[DuckdbTable]:
        """Secondary tables used by the feature (via SQL queries).

        This affects the data available via the connection passed to compute(); only the tables and columns
        declared here or in `self.join_strategies` are guaranteed to exist,
        and only they may be accessed by compute.
        """
        ...

    @property
    @abstractmethod
    def join_strategies(self) -> Sequence[JoinStrategy]:
        """Join strategies used by the feature via python methods on the strategies.

        This affects the data available via the connection passed to compute(); only the tables and columns
        used by these strategies or declared in `self.secondary_tables` are guaranteed to exist,
        and only they may be accessed by compute.
        """
        ...

    @abstractmethod
    def is_numeric(self) -> bool: ...

    # A feature must override at least one of acompute or acompute_batch.

    async def acompute(self, args: tuple[Any, ...],
                       conn: DuckDBPyConnection) -> T | None:
        """Compute the feature on a single row.

        The arguments `args` are in the order given by `self.params`.

        The default implementation delegates to compute_batch and is quite inefficient;
        if you override the batch implementation, please consider if you can also override this one
        more efficiently.

        All secondary tables are available in the provided `conn`ection.
        """
        df = pl.DataFrame(
            {col.name: [value] for col, value in zip(self.params.cols, args, strict=True)},
            schema=self.params.to_polars()
        )
        return (await self.acompute_batch(Dataset(self.params, df), conn))[0]

    async def acompute_safe(self, args: tuple[Any, ...],
                            conn: DuckDBPyConnection) -> T | None:
        """As `acompute`, but returns None (a missing value) instead of raising an error.

        For categorical features, also returns the Other category if `compute` returned an unexpected value.
        """
        try:
            return await self.acompute(args, conn)
        except Exception: # noqa: BLE001
            return None

    async def acompute_with_defaults(self, args: tuple[Any, ...],
                                     conn: DuckDBPyConnection) -> T:
        """As `acompute`, but substitutes the self.default_for_xxx values in case of missing values or errors."""
        value = await self.acompute_safe(args, conn)
        return self.substitute_defaults(value)

    def substitute_defaults(self, value: T | None) -> T:
        """Apply the same logic as acompute_with_defaults.

        This method should NOT be overridden by feature implementations.
        """
        if value is None:
            return self.default_for_missing
        return value

    def substitute_defaults_batch(self, values: pl.Series) -> pl.Series:
        """Apply the same logic as acompute_batch_with_defaults.

        This method should NOT be overridden by feature implementations.
        """
        return values.fill_null(self.default_for_missing)

    async def acompute_batch(self, input: Dataset,
                             conn: DuckDBPyConnection) -> pl.Series:
        """The default implementation delegates to acompute (non-batch version).

        If that raises an error for some rows, those rows get missing values in the output series.
        However, a 'real' batch implementation overriding this method is allowed to fail the entire batch
        by propagating the error, even if it might have succeeded for a subset of the rows.
        """
        strict_df = pl.DataFrame([input.data.get_column(col.name) for col in self.params.cols])
        results = await amap_gather_with_limit(strict_df.iter_rows(), lambda row: self.acompute_safe(row, conn), True)
        results = [None if isinstance(result, BaseException) else result for result in results]
        return pl.Series(name=self.name, dtype=self.raw_dtype.polars_type, values=results)

    async def acompute_batch_safe(self, input: Dataset,
                                  conn: DuckDBPyConnection) -> pl.Series:
        try:
            return (await self.acompute_batch(input, conn)).cast(self.dtype.polars_type, strict=False).rename(self.name)
        except Exception: # noqa: BLE001
            return pl.repeat(None, len(input), dtype=self.dtype.polars_type, eager=True).rename(self.name)

    async def acompute_batch_with_defaults(self, input: Dataset,
                                           conn: DuckDBPyConnection) -> pl.Series:
        """As `acompute_batch`, but substitutes the self.default_for_xxx values for missing values.

        An error causes the whole batch's output to be returned as the default for errors.

        This method should NOT be overridden by feature implementations.
        """
        return self.substitute_defaults_batch(await self.acompute_batch_safe(input, conn))

# Every feature must implement exactly one of the feature value type interfaces (IntFeature, etc) - 
# it is not enough to directly implement e.g. Feature[int].

# -------- Feature value types

# Features of all types can return missing values. The type param T is the non-missing type (e.g. int) and not the
# returned type (e.g. int | None).
# Generally speaking, features should only return missing values if one of the inputs has a missing value.

class NumericFeature[T](Feature[T]):
    @final
    @override
    def is_numeric(self) -> bool: return True

@define(slots=False)
class IntFeature(NumericFeature[int]):
    @final
    @property
    @override
    def dtype(self) -> Dtype: return agentune.analyze.core.types.int32

@define(slots=False)
class FloatFeature(NumericFeature[float]):
    # Redeclare with concrete types to work around attrs issue
    default_for_missing: float
    default_for_nan: float
    default_for_infinity: float
    default_for_neg_infinity: float

    @final
    @property
    @override
    def dtype(self) -> Dtype: return agentune.analyze.core.types.float64

    @final
    @override
    def substitute_defaults(self, result: float | None) -> float:
        if result is None: return self.default_for_missing
        elif result == math.inf: return self.default_for_infinity
        elif result == (- math.inf): return self.default_for_neg_infinity
        elif math.isnan(result): return self.default_for_nan
        else: return result

    @final
    @override
    def substitute_defaults_batch(self, series: pl.Series) -> pl.Series:
        return series.replace([None, math.nan, math.inf, -math.inf],
                              [self.default_for_missing, self.default_for_nan, self.default_for_infinity, self.default_for_neg_infinity])


@define(slots=False)
class BoolFeature(Feature[bool]):
    @final
    @override
    def is_numeric(self) -> bool: return False

    @final
    @property
    @override
    def dtype(self) -> Dtype: return agentune.analyze.core.types.boolean

@define(slots=False)
class CategoricalFeature(Feature[str]):
    """Categorical features output scalar strings, but the column type (in compute_batch_safe) is the enum dtype
    corresponding to the feature's list of categories, with other_category at the end.
    """

    # Special category name that every categorical feature is allowed to return if it encounters an unexpected value.
    other_category: ClassVar[str] = '_other_'

    # Possible categories of this feature, not including the special other_category.
    categories: tuple[str, ...] = attrs.field()

    @categories.validator
    def _categories_validator(self, _attribute: attrs.Attribute, value: tuple[str, ...]) -> None:
        if len(value) == 0:
            raise ValueError('CategoricalFeature must have at least one category')
        if self.other_category in value:
            raise ValueError(f'CategoricalFeature cannot contain the special Other category {CategoricalFeature.other_category}')
        if '' in value:
            raise ValueError('The empty string is not a valid category')
        if len(set(value)) != len(value):
            raise ValueError('CategoricalFeature cannot contain duplicate categories')

    @final
    @property
    def categories_with_other(self) -> Sequence[str]:
        return (*self.categories, CategoricalFeature.other_category)
    
    @final
    @override
    def is_numeric(self) -> bool: return False

    @final
    @property
    @override
    def dtype(self) -> Dtype:
        return agentune.analyze.core.types.EnumDtype(*self.categories, CategoricalFeature.other_category)

    @final
    @property
    @override
    def raw_dtype(self) -> Dtype:
        return agentune.analyze.core.types.string

    @override
    async def acompute_batch(self, input: Dataset,
                             conn: DuckDBPyConnection) -> pl.Series:
        """The default implementation delegates to acompute (non-batch version).

        If that raises an error for some rows, those rows get missing values in the output series.
        However, a 'real' batch implementation overriding this method is allowed to fail the entire batch
        by propagating the error, even if it might have succeeded for a subset of the rows.
        """
        # Unlike the super default, we want to preserve strings that are not in the categories list
        # and not yet replace them with other_category; we only replace errors with missing values,
        # to be as similar as possible to a feature that overrides this method.
        async def acompute_error_to_none(row: tuple[Any, ...]) -> str | None:
            try:
                return await self.acompute(row, conn)
            except Exception: # noqa: BLE001
                return None
        strict_df = pl.DataFrame([input.data.get_column(col.name) for col in self.params.cols])
        results = await amap_gather_with_limit(strict_df.iter_rows(), acompute_error_to_none, False)
        return pl.Series(name=self.name, dtype=self.raw_dtype.polars_type, values=results)

    @final
    @override
    async def acompute_safe(self, args: tuple[Any, ...],
                            conn: DuckDBPyConnection) -> str | None:
        result = await super().acompute_safe(args, conn)
        if result == '':
            return None
        elif result is not None and result != CategoricalFeature.other_category and result not in self.categories:
            return CategoricalFeature.other_category
        else:
            return result

    def _series_result_with_other_category(self, series: pl.Series) -> pl.Series:
        if series.dtype == pl.datatypes.String:
            series = series.replace('', None)
        df = pl.DataFrame({'raw': series})
        return df.select(
            pl.when(pl.col('raw').cast(self.dtype.polars_type, strict=False).is_null() & pl.col('raw').is_not_null()) \
              .then(pl.lit(CategoricalFeature.other_category)) \
              .otherwise(pl.col('raw')) \
              .cast(self.dtype.polars_type)
              .alias(self.name)
        )[self.name]

    @override
    async def acompute_batch_safe(self, input: Dataset,
                                  conn: DuckDBPyConnection) -> pl.Series:
        # Don't call super().acompute_batch_safe; we want to transform unexpected values into other_category
        # before casting the result to the enum type (which would transform them to missing values),
        # which means we use self.raw_dtype where super().acompute_batch_safe uses self.dtype
        try:
            result = (await self.acompute_batch(input, conn)).cast(self.raw_dtype.polars_type, strict=False).rename(self.name)
        except Exception: # noqa: BLE001
            return pl.repeat(None, len(input), dtype=self.dtype.polars_type, eager=True).rename(self.name)
        return self._series_result_with_other_category(result)

# -------- Synchronous features
# A synchronous feature must extend one of the subclasses specific to the feature type, like SyncIntFeature.

class SyncFeature[T](Feature[T]):
    # A feature must override at least one of compute or compute_batch
    
    def compute(self, args: tuple[Any, ...],
                conn: DuckDBPyConnection) -> T | None:
        df = pl.DataFrame(
            {col.name: [value] for col, value in zip(self.params.cols, args, strict=True)},
            schema=self.params.to_polars()
        )
        return self.compute_batch(Dataset(self.params, df), conn)[0]

    def compute_safe(self, args: tuple[Any, ...],
                     conn: DuckDBPyConnection) -> T | None:
        try:
            return self.compute(args, conn)
        except Exception: # noqa: BLE001
            return None

    @final
    def compute_with_defaults(self, args: tuple[Any, ...],
                              conn: DuckDBPyConnection) -> T:
        value = self.compute_safe(args, conn)
        return self.substitute_defaults(value)

    def compute_batch(self, input: Dataset,
                      conn: DuckDBPyConnection) -> pl.Series:
        """The default implementation delegates to compute (non-batch version).

        If that raises an error for some rows, those rows get missing values in the output series.
        However, a 'real' batch implementation overriding this method is allowed to fail the entire batch
        by propagating the error, even if it might have succeeded for a subset of the rows.
        """
        strict_df = pl.DataFrame([input.data.get_column(col.name) for col in self.params.cols])
        return pl.Series(name=self.name, dtype=self.raw_dtype.polars_type,
                         values=[self.compute_safe(row, conn) for row in strict_df.iter_rows()])

    def compute_batch_safe(self, input: Dataset,
                           conn: DuckDBPyConnection) -> pl.Series:
        try:
            return self.compute_batch(input, conn).cast(self.dtype.polars_type, strict=False).rename(self.name)
        except Exception: # noqa: BLE001
            return pl.repeat(None, len(input), dtype=self.dtype.polars_type, eager=True).rename(self.name)

    @final
    def compute_batch_with_defaults(self, input: Dataset,
                                    conn: DuckDBPyConnection) -> pl.Series:
        return self.substitute_defaults_batch(self.compute_batch_safe(input, conn))

    @override 
    async def acompute(self, args: tuple[Any, ...],
                       conn: DuckDBPyConnection) -> T | None:
        with conn.cursor() as cursor:
            return await asyncio.to_thread(self.compute, args, cursor)

    @override
    async def acompute_safe(self, args: tuple[Any, ...],
                            conn: DuckDBPyConnection) -> T | None:
        with conn.cursor() as cursor:
            return await asyncio.to_thread(self.compute_safe, args, cursor)

    @override
    async def acompute_with_defaults(self, args: tuple[Any, ...],
                                     conn: DuckDBPyConnection) -> T:
        with conn.cursor() as cursor:
            return await asyncio.to_thread(self.compute_with_defaults, args, cursor)

    @override
    async def acompute_batch(self, input: Dataset,
                             conn: DuckDBPyConnection) -> pl.Series:
        with conn.cursor() as cursor:
            return await asyncio.to_thread(self.compute_batch, input, cursor)

    @override
    async def acompute_batch_safe(self, input: Dataset,
                                  conn: DuckDBPyConnection) -> pl.Series:
        with conn.cursor() as cursor:
            return await asyncio.to_thread(self.compute_batch_safe, input, cursor)


    @override
    async def acompute_batch_with_defaults(self, input: Dataset,
                                           conn: DuckDBPyConnection) -> pl.Series:
        with conn.cursor() as cursor:
            return await asyncio.to_thread(self.compute_batch_with_defaults, input, cursor)

class SyncIntFeature(IntFeature, SyncFeature[int]): pass

class SyncFloatFeature(FloatFeature, SyncFeature[float]): pass

class SyncBoolFeature(BoolFeature, SyncFeature[bool]): pass

class SyncCategoricalFeature(CategoricalFeature, SyncFeature[str]):
    @override
    def compute_safe(self, args: tuple[Any, ...],
                     conn: DuckDBPyConnection) -> str | None:
        result = super().compute_safe(args, conn)
        if result == '':
            return None
        elif result is not None and result != CategoricalFeature.other_category and result not in self.categories:
            return CategoricalFeature.other_category
        else:
            return result


    @override
    def compute_batch(self, input: Dataset,
                      conn: DuckDBPyConnection) -> pl.Series:
        """The default implementation delegates to compute (non-batch version).

        If that raises an error for some rows, those rows get missing values in the output series.
        However, a 'real' batch implementation overriding this method is allowed to fail the entire batch
        by propagating the error, even if it might have succeeded for a subset of the rows.
        """
        # Unlike the super default, we want to preserve strings that are not in the categories list
        # and not yet replace them with other_category; we only replace errors with missing values,
        # to be as similar as possible to a feature that overrides this method.
        def compute_error_to_none(row: tuple[Any, ...]) -> str | None:
            try:
                return self.compute(row, conn)
            except Exception: # noqa: BLE001
                return None
        strict_df = pl.DataFrame([input.data.get_column(col.name) for col in self.params.cols])
        return pl.Series(name=self.name, dtype=self.raw_dtype.polars_type,
                         values=[compute_error_to_none(row) for row in strict_df.iter_rows()])

    @override
    def compute_batch_safe(self, input: Dataset,
                           conn: DuckDBPyConnection) -> pl.Series:
        # Don't call super().acompute_batch_safe; we want to transform unexpected values into other_category
        # before casting the result to the enum type (which would transform them to missing values),
        # which means we use self.raw_dtype where super().acompute_batch_safe uses self.dtype
        try:
            result = self.compute_batch(input, conn).cast(self.raw_dtype.polars_type, strict=False).rename(self.name)
        except Exception: # noqa: BLE001
            return pl.repeat(None, len(input), dtype=self.dtype.polars_type, eager=True).rename(self.name)
        return self._series_result_with_other_category(result)

# -------- Other feature types used as public APIs

class SqlQueryFeature(Feature):
    """A feature that can be represented to the user as an SQL query.

    Extending this class doesn't necessarily mean that a feature is implemented as an SQL query.
    """

    @property
    @abstractmethod
    def sql_query(self) -> str: ...

class WrappedFeature(Feature):
    """A feature which wraps another, e.g. converting a numeric feature to a boolean one by applying a cutoff."""

    @property
    @abstractmethod
    def inner(self) -> Feature: ...

# This is an example; it may not prove useful, and can be removed. 
# The important thing is that a feature using an LLM should have a parameter of type LLMWithSpec.

class LlmFeature[T](Feature[T]):
    """A feature that is computed by an LLM."""

    @property
    @abstractmethod
    def model(self) -> LLMWithSpec: ...
