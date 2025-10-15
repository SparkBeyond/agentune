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
from agentune.analyze.feature.eval.limits import amap_gather_with_limit
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
        The methods (a)evaluate, (a)evaluate_batch can raise an error, return a missing value (None for `evaluate`),
        return a NaN or +/- infinity value for float features, and return an unexpected string for categorical features.

        The _safe variants return None instead of raising an error.
        For categorical features, they also return the special value `CategoricalFeature.other_category`
        if `evaluate` returns an unexpected value (one not in the feature's categories list).

        The _with_defaults variants substitute the default_for_xxx attributes in these five cases.

    Implementation note:
        This base class is annotated with @attrs.define, and all implementations must be attrs classes.
        We rely on attrs.evolve() being able to change e.g. feature names and descriptions.
        Only attributes that must be free parameters to the feature are explicitly declared as attributes.

    Args:
        name: Used as the column/series name in outputs. Not guaranteed to be unique among Feature instances.
        description: Human-readable description of the feature.
        technical_description: Human-readable description of feature's implementation details.
        default_for_missing: a value substituted by evaluate_with_defaults if the underlying `evaluate` outputs
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
        """The dtype of series returned by aevaluate_batch_safe. See also raw_dtype."""
        ...

    @property
    def raw_dtype(self) -> Dtype:
        """The dtype of series returned by aevaluate_batch (the non-safe version).

        Can be more general than `self.dtype`, with the _safe evaluation coercing raw values to the right dtype.
        """
        return self.dtype

    @property
    @abstractmethod
    def params(self) -> Schema: 
        """Columns of the main table used by the feature.
        This affects the parameters to evaluate().
        """
        ...
    
    @property
    @abstractmethod
    def secondary_tables(self) -> Sequence[DuckdbTable]:
        """Secondary tables used by the feature (via SQL queries).

        This affects the data available via the connection passed to evaluate(); only the tables and columns
        declared here or in `self.join_strategies` are guaranteed to exist,
        and only they may be accessed by evaluate.
        """
        ...

    @property
    @abstractmethod
    def join_strategies(self) -> Sequence[JoinStrategy]:
        """Join strategies used by the feature via python methods on the strategies.

        This affects the data available via the connection passed to evaluate(); only the tables and columns
        used by these strategies or declared in `self.secondary_tables` are guaranteed to exist,
        and only they may be accessed by evaluate.
        """
        ...

    @abstractmethod
    def is_numeric(self) -> bool: ...

    # A feature must override at least one of evaluate or evaluate_batch.

    async def aevaluate(self, args: tuple[Any, ...], 
                        conn: DuckDBPyConnection) -> T | None:
        """Evaluate a single row.

        The arguments `args` are in the order given by `self.params`.

        The default implementation delegates to evaluate_batch and is quite inefficient;
        if you override the batch implementation, please consider if you can also override this one
        more efficiently.

        All secondary tables are available in the provided `conn`ection.
        """
        df = pl.DataFrame(
            {col.name: [value] for col, value in zip(self.params.cols, args, strict=True)},
            schema=self.params.to_polars()
        )
        return (await self.aevaluate_batch(Dataset(self.params, df), conn))[0]

    async def aevaluate_safe(self, args: tuple[Any, ...], 
                             conn: DuckDBPyConnection) -> T | None:
        """As `aevaluate`, but returns None (a missing value) instead of raising an error.

        For categorical features, also returns the Other category if `evaluate` returned an unexpected value.
        """
        try:
            return await self.aevaluate(args, conn)
        except Exception: # noqa: BLE001
            return None

    async def aevaluate_with_defaults(self, args: tuple[Any, ...], 
                                      conn: DuckDBPyConnection) -> T:
        """As `aevaluate`, but substitutes the self.default_for_xxx values in case of missing values or errors."""
        value = await self.aevaluate_safe(args, conn)
        return self.substitute_defaults(value)

    def substitute_defaults(self, value: T | None) -> T:
        """Apply the same logic as aevaluate_with_defaults.

        This method should NOT be overridden by feature implementations.
        """
        if value is None:
            return self.default_for_missing
        return value

    def substitute_defaults_batch(self, values: pl.Series) -> pl.Series:
        """Apply the same logic as aevaluate_batch_with_defaults.

        This method should NOT be overridden by feature implementations.
        """
        return values.fill_null(self.default_for_missing)

    async def aevaluate_batch(self, input: Dataset, 
                              conn: DuckDBPyConnection) -> pl.Series:
        """The default implementation delegates to aevaluate (non-batch version).

        If that raises an error for some rows, those rows get missing values in the output series.
        However, a 'real' batch implementation overriding this method is allowed to fail the entire batch
        by propagating the error, even if it might have succeeded for a subset of the rows.
        """
        strict_df = pl.DataFrame([input.data.get_column(col.name) for col in self.params.cols])
        results = await amap_gather_with_limit(strict_df.iter_rows(), lambda row: self.aevaluate_safe(row, conn), True)
        results = [None if isinstance(result, BaseException) else result for result in results]
        return pl.Series(name=self.name, dtype=self.raw_dtype.polars_type, values=results)

    async def aevaluate_batch_safe(self, input: Dataset, 
                                   conn: DuckDBPyConnection) -> pl.Series:
        try:
            return (await self.aevaluate_batch(input, conn)).cast(self.dtype.polars_type, strict=False).rename(self.name)
        except Exception: # noqa: BLE001
            return pl.repeat(None, len(input), dtype=self.dtype.polars_type, eager=True).rename(self.name)

    async def aevaluate_batch_with_defaults(self, input: Dataset,
                                            conn: DuckDBPyConnection) -> pl.Series:
        """As `aevaluate_batch`, but substitutes the self.default_for_xxx values for missing values.

        An error causes the whole batch's output to be returned as the default for errors.

        This method should NOT be overridden by feature implementations.
        """
        return self.substitute_defaults_batch(await self.aevaluate_batch_safe(input, conn))

# Every feature must implement exactly one of the feature value type interfaces (IntFeature, etc) - 
# it is not enough to directly implement e.g. Feature[int].

# -------- Feature value types

# Note that the values of type param T are all non-None; missing values are allowed in feature outputs.
# Generally speaking, features should only return missing values if one of the inputs has a missing value.
# We might prefer a simpler world where feature outputs can't be missing, but that would not allow us
# to use input columns as features.

class NumericFeature[T](Feature[T]):
    @final
    @override
    def is_numeric(self) -> bool: return True

# Other int sizes or unsigned ints can be added as needed.
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
    """Categorical features output scalar strings, but the column type (in evaluate_batch_safe) is the enum dtype
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
    async def aevaluate_batch(self, input: Dataset, 
                              conn: DuckDBPyConnection) -> pl.Series:
        """The default implementation delegates to aevaluate (non-batch version).

        If that raises an error for some rows, those rows get missing values in the output series.
        However, a 'real' batch implementation overriding this method is allowed to fail the entire batch
        by propagating the error, even if it might have succeeded for a subset of the rows.
        """
        # Unlike the super default, we want to preserve strings that are not in the categories list
        # and not yet replace them with other_category; we only replace errors with missing values,
        # to be as similar as possible to a feature that overrides this method.
        async def aevaluate_error_to_none(row: tuple[Any, ...]) -> str | None:
            try:
                return await self.aevaluate(row, conn)
            except Exception: # noqa: BLE001
                return None
        strict_df = pl.DataFrame([input.data.get_column(col.name) for col in self.params.cols])
        results = await amap_gather_with_limit(strict_df.iter_rows(), aevaluate_error_to_none, False)
        return pl.Series(name=self.name, dtype=self.raw_dtype.polars_type, values=results)

    @final
    @override
    async def aevaluate_safe(self, args: tuple[Any, ...], 
                             conn: DuckDBPyConnection) -> str | None:
        result = await super().aevaluate_safe(args, conn)
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
    async def aevaluate_batch_safe(self, input: Dataset, 
                                   conn: DuckDBPyConnection) -> pl.Series:
        # Don't call super().aevaluate_batch_safe; we want to transform unexpected values into other_category
        # before casting the result to the enum type (which would transform them to missing values),
        # which means we use self.raw_dtype where super().aevaluate_batch_safe uses self.dtype
        try:
            result = (await self.aevaluate_batch(input, conn)).cast(self.raw_dtype.polars_type, strict=False).rename(self.name)
        except Exception: # noqa: BLE001
            return pl.repeat(None, len(input), dtype=self.dtype.polars_type, eager=True).rename(self.name)
        return self._series_result_with_other_category(result)

# -------- Synchronous features
# A synchronous feature must extend one of the subclasses specific to the feature type, like SyncIntFeature.

class SyncFeature[T](Feature[T]):
    # A feature must override at least one of evaluate or evaluate_batch.
    
    def evaluate(self, args: tuple[Any, ...], 
                 conn: DuckDBPyConnection) -> T | None:
        df = pl.DataFrame(
            {col.name: [value] for col, value in zip(self.params.cols, args, strict=True)},
            schema=self.params.to_polars()
        )
        return self.evaluate_batch(Dataset(self.params, df), conn)[0]

    def evaluate_safe(self, args: tuple[Any, ...], 
                      conn: DuckDBPyConnection) -> T | None:
        try:
            return self.evaluate(args, conn)
        except Exception: # noqa: BLE001
            return None

    @final
    def evaluate_with_defaults(self, args: tuple[Any, ...],
                               conn: DuckDBPyConnection) -> T:
        value = self.evaluate_safe(args, conn)
        return self.substitute_defaults(value)

    def evaluate_batch(self, input: Dataset,
                       conn: DuckDBPyConnection) -> pl.Series:
        """The default implementation delegates to evaluate (non-batch version).

        If that raises an error for some rows, those rows get missing values in the output series.
        However, a 'real' batch implementation overriding this method is allowed to fail the entire batch
        by propagating the error, even if it might have succeeded for a subset of the rows.
        """
        strict_df = pl.DataFrame([input.data.get_column(col.name) for col in self.params.cols])
        return pl.Series(name=self.name, dtype=self.raw_dtype.polars_type,
                         values=[self.evaluate_safe(row, conn) for row in strict_df.iter_rows()])

    def evaluate_batch_safe(self, input: Dataset,
                            conn: DuckDBPyConnection) -> pl.Series:
        try:
            return self.evaluate_batch(input, conn).cast(self.dtype.polars_type, strict=False).rename(self.name)
        except Exception: # noqa: BLE001
            return pl.repeat(None, len(input), dtype=self.dtype.polars_type, eager=True).rename(self.name)

    @final
    def evaluate_batch_with_defaults(self, input: Dataset,
                                     conn: DuckDBPyConnection) -> pl.Series:
        return self.substitute_defaults_batch(self.evaluate_batch_safe(input, conn))

    @override 
    async def aevaluate(self, args: tuple[Any, ...], 
                       conn: DuckDBPyConnection) -> T | None:
        with conn.cursor() as cursor:
            return await asyncio.to_thread(self.evaluate, args, cursor)

    @override
    async def aevaluate_safe(self, args: tuple[Any, ...], 
                             conn: DuckDBPyConnection) -> T | None:
        with conn.cursor() as cursor:
            return await asyncio.to_thread(self.evaluate_safe, args, cursor)

    @override
    async def aevaluate_with_defaults(self, args: tuple[Any, ...], 
                                      conn: DuckDBPyConnection) -> T:
        with conn.cursor() as cursor:
            return await asyncio.to_thread(self.evaluate_with_defaults, args, cursor)

    @override
    async def aevaluate_batch(self, input: Dataset, 
                              conn: DuckDBPyConnection) -> pl.Series:
        with conn.cursor() as cursor:
            return await asyncio.to_thread(self.evaluate_batch, input, cursor)

    @override
    async def aevaluate_batch_safe(self, input: Dataset, 
                                   conn: DuckDBPyConnection) -> pl.Series:
        with conn.cursor() as cursor:
            return await asyncio.to_thread(self.evaluate_batch_safe, input, cursor)


    @override
    async def aevaluate_batch_with_defaults(self, input: Dataset, 
                                            conn: DuckDBPyConnection) -> pl.Series:
        with conn.cursor() as cursor:
            return await asyncio.to_thread(self.evaluate_batch_with_defaults, input, cursor)

class SyncIntFeature(IntFeature, SyncFeature[int]): pass

class SyncFloatFeature(FloatFeature, SyncFeature[float]): pass

class SyncBoolFeature(BoolFeature, SyncFeature[bool]): pass

class SyncCategoricalFeature(CategoricalFeature, SyncFeature[str]):
    @override
    def evaluate_safe(self, args: tuple[Any, ...], 
                      conn: DuckDBPyConnection) -> str | None:
        result = super().evaluate_safe(args, conn)
        if result == '':
            return None
        elif result is not None and result != CategoricalFeature.other_category and result not in self.categories:
            return CategoricalFeature.other_category
        else:
            return result


    @override
    def evaluate_batch(self, input: Dataset, 
                       conn: DuckDBPyConnection) -> pl.Series:
        """The default implementation delegates to evaluate (non-batch version).

        If that raises an error for some rows, those rows get missing values in the output series.
        However, a 'real' batch implementation overriding this method is allowed to fail the entire batch
        by propagating the error, even if it might have succeeded for a subset of the rows.
        """
        # Unlike the super default, we want to preserve strings that are not in the categories list
        # and not yet replace them with other_category; we only replace errors with missing values,
        # to be as similar as possible to a feature that overrides this method.
        def evaluate_error_to_none(row: tuple[Any, ...]) -> str | None:
            try:
                return self.evaluate(row, conn)
            except Exception: # noqa: BLE001
                return None
        strict_df = pl.DataFrame([input.data.get_column(col.name) for col in self.params.cols])
        return pl.Series(name=self.name, dtype=self.raw_dtype.polars_type,
                         values=[evaluate_error_to_none(row) for row in strict_df.iter_rows()])

    @override
    def evaluate_batch_safe(self, input: Dataset, 
                                   conn: DuckDBPyConnection) -> pl.Series:
        # Don't call super().aevaluate_batch_safe; we want to transform unexpected values into other_category
        # before casting the result to the enum type (which would transform them to missing values),
        # which means we use self.raw_dtype where super().aevaluate_batch_safe uses self.dtype
        try:
            result = self.evaluate_batch(input, conn).cast(self.raw_dtype.polars_type, strict=False).rename(self.name)
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
    """A feature that is evaluated by an LLM."""

    @property
    @abstractmethod
    def model(self) -> LLMWithSpec: ...
