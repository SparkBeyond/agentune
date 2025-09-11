import asyncio
import logging
import math
from collections.abc import Sequence
from typing import Any, override

import polars as pl
import pytest
from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.feature.base import (
    BoolFeature,
    CategoricalFeature,
    Feature,
    FloatFeature,
    IntFeature,
    SyncBoolFeature,
    SyncCategoricalFeature,
    SyncFeature,
    SyncFloatFeature,
    SyncIntFeature,
)
from agentune.analyze.join.base import JoinStrategy

_logger = logging.getLogger(__name__)


class BatchingTestFeature[T](Feature[T]):
    """Simulating a feature that overrides aevaluate_batch and fails the whole batch if any row fails."""

    @override
    async def aevaluate_batch(self, input: Dataset, 
                              conn: DuckDBPyConnection) -> pl.Series:
        # As in the base version, but calling self.aevaluate not self.aevalute_safe
        strict_df = pl.DataFrame([input.data.get_column(col.name) for col in self.params.cols])
        results = await asyncio.gather(*[self.aevaluate(row, conn) for row in strict_df.iter_rows()])
        return pl.Series(name=self.name, dtype=self.raw_dtype.polars_type, values=results)

class BatchingSyncTestFeature[T](SyncFeature[T]):
    """Simulating a feature that overrides aevaluate_batch and fails the whole batch if any row fails."""

    @override
    def evaluate_batch(self, input: Dataset, 
                       conn: DuckDBPyConnection) -> pl.Series:
        # As in the base version, but calling self.evaluate not self.evalute_safe
        strict_df = pl.DataFrame([input.data.get_column(col.name) for col in self.params.cols])
        return pl.Series(name=self.name, dtype=self.raw_dtype.polars_type,
                         values=[self.evaluate(row, conn) for row in strict_df.iter_rows()])


@frozen
class IntTestFeature(IntFeature):
    # Redeclare everything with defaults
    name: str = 'IntFeature'
    description: str = ''
    technical_description: str = ''

    default_for_missing: int = 2

    params: Schema = Schema((Field('int', types.int32),))
    secondary_tables: Sequence[DuckdbTable] = ()
    join_strategies: Sequence[JoinStrategy] = ()

    @override
    async def aevaluate(self, args: tuple[Any, ...], 
                        conn: DuckDBPyConnection) -> int | None:
        match args[0]:
            case None: return -1
            case 0: return None
            case 1: raise ValueError('1')
            case int(other): return other
            case _: raise ValueError('other')

@frozen
class SyncIntTestFeature(SyncIntFeature):
    # Redeclare everything with defaults
    name: str = 'IntFeature'
    description: str = ''
    technical_description: str = ''

    default_for_missing: int = 2

    params: Schema = Schema((Field('int', types.int32),))
    secondary_tables: Sequence[DuckdbTable] = ()
    join_strategies: Sequence[JoinStrategy] = ()

    @override
    def evaluate(self, args: tuple[Any, ...], 
                 conn: DuckDBPyConnection) -> int | None:
        match args[0]:
            case None: return -1
            case 0: return None
            case 1: raise ValueError('1')
            case int(other): return other
            case _: raise ValueError('other')

@frozen
class BoolTestFeature(BoolFeature):
    # Redeclare everything with defaults
    name: str = 'BoolFeature'
    description: str = ''
    technical_description: str = ''

    default_for_missing: bool = True

    params: Schema = Schema((Field('int', types.int32),))
    secondary_tables: Sequence[DuckdbTable] = ()
    join_strategies: Sequence[JoinStrategy] = ()

    @override
    async def aevaluate(self, args: tuple[Any, ...], 
                        conn: DuckDBPyConnection) -> bool | None:
        match args[0]:
            case None: return False
            case 1: return None
            case 2: return True
            case 3: raise ValueError('1')
            case _: raise ValueError('other')

@frozen
class SyncBoolTestFeature(SyncBoolFeature):
    # Redeclare everything with defaults
    name: str = 'BoolFeature'
    description: str = ''
    technical_description: str = ''

    default_for_missing: bool = True

    params: Schema = Schema((Field('int', types.int32),))
    secondary_tables: Sequence[DuckdbTable] = ()
    join_strategies: Sequence[JoinStrategy] = ()

    @override
    def evaluate(self, args: tuple[Any, ...], 
                 conn: DuckDBPyConnection) -> bool | None:
        match args[0]:
            case None: return False
            case 1: return None
            case 2: return True
            case 3: raise ValueError('1')
            case _: raise ValueError('other')


@frozen
class FloatTestFeature(FloatFeature):
    # Redeclare everything with defaults
    name: str = 'FloatFeature'
    description: str = ''
    technical_description: str = ''

    default_for_missing: float = 10.0
    default_for_nan: float = 11.0
    default_for_infinity: float = 12.0
    default_for_neg_infinity: float = 13.0

    params: Schema = Schema((Field('int', types.int32),))
    secondary_tables: Sequence[DuckdbTable] = ()
    join_strategies: Sequence[JoinStrategy] = ()

    @override
    async def aevaluate(self, args: tuple[Any, ...], 
                        conn: DuckDBPyConnection) -> float | None:
        match args[0]:
            case None: return -1.0
            case 0: return None
            case 1: raise ValueError('1')
            case 2: return math.nan
            case 3: return math.inf
            case 4: return -math.inf
            case int(other): return float(other)
            case _: raise ValueError('other')


@frozen
class SyncFloatTestFeature(SyncFloatFeature):
    # Redeclare everything with defaults
    name: str = 'FloatFeature'
    description: str = ''
    technical_description: str = ''

    default_for_missing: float = 10.0
    default_for_nan: float = 11.0
    default_for_infinity: float = 12.0
    default_for_neg_infinity: float = 13.0

    params: Schema = Schema((Field('int', types.int32),))
    secondary_tables: Sequence[DuckdbTable] = ()
    join_strategies: Sequence[JoinStrategy] = ()

    @override
    def evaluate(self, args: tuple[Any, ...], 
                 conn: DuckDBPyConnection) -> float | None:
        match args[0]:
            case None: return -1.0
            case 0: return None
            case 1: raise ValueError('1')
            case 2: return math.nan
            case 3: return math.inf
            case 4: return -math.inf
            case int(other): return float(other)
            case _: raise ValueError('other')

@frozen
class CategoricalTestFeature(CategoricalFeature):
    # Redeclare everything with defaults
    name: str = 'CategoricalFeature'
    description: str = ''
    technical_description: str = ''

    categories: tuple[str, ...] = ('a', 'b', 'c')

    default_for_missing: str = 'c'

    params: Schema = Schema((Field('int', types.int32),))
    secondary_tables: Sequence[DuckdbTable] = ()
    join_strategies: Sequence[JoinStrategy] = ()

    @override
    async def aevaluate(self, args: tuple[Any, ...], 
                        conn: DuckDBPyConnection) -> str | None:
        match args[0]:
            case None: return 'a'
            case 0: return None
            case 1: raise ValueError('1')
            case 2: return 'b'
            case 3: return 'c'
            case 4: return CategoricalFeature.other_category
            case 5: return 'd' # not a valid category!
            case 6: return ''
            case _: raise ValueError('other')

@frozen
class SyncCategoricalTestFeature(SyncCategoricalFeature):
    # Redeclare everything with defaults
    name: str = 'CategoricalFeature'
    description: str = ''
    technical_description: str = ''

    categories: tuple[str, ...] = ('a', 'b', 'c')

    default_for_missing: str = 'c'

    params: Schema = Schema((Field('int', types.int32),))
    secondary_tables: Sequence[DuckdbTable] = ()
    join_strategies: Sequence[JoinStrategy] = ()

    @override
    def evaluate(self, args: tuple[Any, ...], 
                 conn: DuckDBPyConnection) -> str | None:
        match args[0]:
            case None: return 'a'
            case 0: return None
            case 1: raise ValueError('1')
            case 2: return 'b'
            case 3: return 'c'
            case 4: return CategoricalFeature.other_category
            case 5: return 'd' # not a valid category!
            case 6: return ''
            case _: raise ValueError('other')


@frozen
class BatchIntTestFeature(IntTestFeature, BatchingTestFeature[int]): pass

@frozen
class BatchSyncIntTestFeature(SyncIntTestFeature, BatchingSyncTestFeature[int]): pass

@frozen
class BatchFloatTestFeature(FloatTestFeature, BatchingTestFeature[float]): pass

@frozen
class BatchSyncFloatTestFeature(SyncFloatTestFeature, BatchingSyncTestFeature[float]): pass

@frozen
class BatchBoolTestFeature(BoolTestFeature, BatchingTestFeature[bool]): pass

@frozen
class BatchSyncBoolTestFeature(SyncBoolTestFeature, BatchingSyncTestFeature[bool]): pass

@frozen
class BatchCategoricalTestFeature(CategoricalTestFeature):
    @override
    async def aevaluate_batch(self, input: Dataset, 
                              conn: DuckDBPyConnection) -> pl.Series:
        # As the base version, but fail the whole batch if any row fails
        strict_df = pl.DataFrame([input.data.get_column(col.name) for col in self.params.cols])
        results = await asyncio.gather(*[self.aevaluate(row, conn) for row in strict_df.iter_rows()])
        return pl.Series(name=self.name, dtype=self.raw_dtype.polars_type, values=results)

@frozen
class BatchSyncCategoricalTestFeature(SyncCategoricalTestFeature):

    @override
    def evaluate_batch(self, input: Dataset, 
                       conn: DuckDBPyConnection) -> pl.Series:
        # As the base version, but fail the whole batch if any row fails
        strict_df = pl.DataFrame([input.data.get_column(col.name) for col in self.params.cols])
        return pl.Series(name=self.name, dtype=self.raw_dtype.polars_type,
                         values=[self.evaluate(row, conn) for row in strict_df.iter_rows()])


async def do_test_feature[T](feature: Feature[T], sync_feature: SyncFeature[T], conn: DuckDBPyConnection,
                             expected_evaluate: dict[int | None, T | None | type[Exception]],
                             safe_substitutions: dict[T | None, T | None],
                             defaults_substitutions: list[tuple[T | None, T]], # Not a dict because it can contain nan key
                             single_error_fails_whole_batch: bool = False) -> None:
    def inputs(ints: list[int | None]) -> Dataset:
        return Dataset.from_polars(pl.DataFrame({'int': pl.Series(values=ints, dtype=pl.Int32)}))

    # Build expected outputs of evaluate_safe and evaluate_with_defaults

    expected_safe: dict[int | None, T | None] = {}
    for k, v in expected_evaluate.items():
        if isinstance(v, type):
            expected_safe[k] = None
        elif v in safe_substitutions:
            expected_safe[k] = safe_substitutions[v]
        else:
            expected_safe[k] = v

    expected_with_defaults: dict[int | None, T] = {}
    for k, v in expected_safe.items():
        for k2, v2 in defaults_substitutions:
            if k2 is v or k2 == v: # 'is' for nan
                expected_with_defaults[k] = v2
                break
        if k not in expected_with_defaults:
            if v is None:
                raise ValueError(f'Missing expected default for key={k} with value={v}')
            expected_with_defaults[k] = v

    non_error_pairs = [(k, v) for k, v in expected_evaluate.items() if v is not ValueError]
    assert len(non_error_pairs) < len(expected_evaluate), 'At least one input should raise an error'
    pairs_with_missing_errors = [(k, None if v is ValueError else v) for k, v in expected_evaluate.items()]
    pairs_with_safe_substitutes = [(k, expected_safe[k]) for k, _v in pairs_with_missing_errors]
    pairs_with_defaults = [(k, expected_with_defaults[k]) for k, _v in pairs_with_missing_errors]

    substitution_for_none = next(v for k, v in defaults_substitutions if k is None)

    # Tests

    for arg, expected_evaluate_result in expected_evaluate.items():
        if expected_evaluate_result is ValueError:
            with pytest.raises(ValueError, match='1'):
                await feature.aevaluate((arg,), conn)
        else:
            actual_evaluate_result = await feature.aevaluate((arg,), conn)
            assert actual_evaluate_result == expected_evaluate_result or actual_evaluate_result is expected_evaluate_result, \
                f'evaluate for {arg} expected {expected_evaluate_result}'

    for arg, expected_safe_result in expected_safe.items():
        actual_evaluate_result = await feature.aevaluate_safe((arg,), conn)
        assert actual_evaluate_result == expected_safe_result or actual_evaluate_result is expected_safe_result, \
            f'evaluate_safe for {arg} expected {expected_safe_result}'

    for arg, expected_defaults_result in expected_with_defaults.items():
        assert await feature.aevaluate_with_defaults((arg,), conn) == expected_defaults_result, \
            f'evaluate_with_defaults for {arg} expected {expected_defaults_result}'

    assert (await feature.aevaluate_batch(inputs([k for k, _v in non_error_pairs]), conn)).equals(
        pl.Series(name=feature.name,
                  values=[v for _k, v in non_error_pairs],
                  dtype=feature.raw_dtype.polars_type),
        check_names=True, check_dtypes=True), 'evaluate_batch (non error outputs)'

    if single_error_fails_whole_batch:
        with pytest.raises(ValueError, match='1'):
            await feature.aevaluate_batch(inputs(list(expected_evaluate.keys())), conn)
    else:
        assert (await feature.aevaluate_batch(inputs(list(expected_evaluate.keys())), conn)).equals(
            pl.Series(name=feature.name,
                      values=[v for _k, v in pairs_with_missing_errors],
                      dtype=feature.raw_dtype.polars_type),
            check_names=True, check_dtypes=True), 'evaluate_batch (substituting None for error outputs)'

    assert (await feature.aevaluate_batch_safe(inputs([k for k, _v in non_error_pairs]), conn)).equals(
        pl.Series(name=feature.name,
                  values=[expected_safe[k] for k, _v in non_error_pairs],
                  dtype=feature.dtype.polars_type),
        check_names=True, check_dtypes=True), 'evaluate_batch_safe (non error outputs)'

    if single_error_fails_whole_batch:
        assert (await feature.aevaluate_batch_safe(inputs(list(expected_evaluate.keys())), conn)).equals(
            pl.Series(name=feature.name,
                      values=[None] * len(expected_evaluate),
                      dtype=feature.dtype.polars_type),
            check_names=True, check_dtypes=True), 'A single error fails the whole batch in evaluate_batch_safe'
    else:
        assert (await feature.aevaluate_batch_safe(inputs(list(expected_evaluate.keys())), conn)).equals(
            pl.Series(name=feature.name,
                      values=[v for _k, v in pairs_with_safe_substitutes],
                      dtype=feature.dtype.polars_type),
            check_names=True, check_dtypes=True), 'Errors cause single-row missing values in evaluate_batch_safe'

    assert (await feature.aevaluate_batch_with_defaults(inputs([k for k, _v in non_error_pairs]), conn)).equals(
        pl.Series(name=feature.name,
                  values=[expected_with_defaults[k] for k, _v in non_error_pairs],
                  dtype=feature.dtype.polars_type),
        check_names=True, check_dtypes=True), 'evaluate_batch_with_defaults (non error outputs)'

    if single_error_fails_whole_batch:
        assert (await feature.aevaluate_batch_with_defaults(inputs(list(expected_evaluate.keys())), conn)).equals(
            pl.Series(name=feature.name,
                      values=[substitution_for_none] * len(expected_evaluate),
                      dtype=feature.dtype.polars_type),
            check_names=True, check_dtypes=True), 'A single error fails the whole batch in evaluate_batch_with_defaults (and substitutes the default value)'
    else:
        assert (await feature.aevaluate_batch_with_defaults(inputs(list(expected_evaluate.keys())), conn)).equals(
            pl.Series(name=feature.name,
                      values=[substitution_for_none if v is None else v for _k, v in pairs_with_defaults],
                      dtype=feature.dtype.polars_type),
            check_names=True, check_dtypes=True), 'Errors cause single-row missing values in evaluate_batch (and the default value is substituted)'

    # Sync tests - keep in sync with above

    for arg, expected_evaluate_result in expected_evaluate.items():
        if expected_evaluate_result is ValueError:
            with pytest.raises(ValueError, match='1'):
                sync_feature.evaluate((arg,), conn)
        else:
            actual_evaluate_result = sync_feature.evaluate((arg,), conn)
            assert actual_evaluate_result == expected_evaluate_result or actual_evaluate_result is expected_evaluate_result, \
                f'evaluate for {arg} expected {expected_evaluate_result}'

    for arg, expected_safe_result in expected_safe.items():
        actual_evaluate_result = sync_feature.evaluate_safe((arg,), conn)
        assert actual_evaluate_result == expected_safe_result or actual_evaluate_result is expected_safe_result, \
            f'evaluate_safe for {arg} expected {expected_safe_result}'

    for arg, expected_defaults_result in expected_with_defaults.items():
        assert sync_feature.evaluate_with_defaults((arg,), conn) == expected_defaults_result, \
            f'evaluate_with_defaults for {arg} expected {expected_defaults_result}'

    assert (sync_feature.evaluate_batch(inputs([k for k, _v in non_error_pairs]), conn)).equals(
        pl.Series(name=sync_feature.name, values=[v for _k, v in non_error_pairs],
                  dtype=sync_feature.raw_dtype.polars_type),
        check_names=True, check_dtypes=True), 'evaluate_batch (non error outputs)'

    if single_error_fails_whole_batch:
        with pytest.raises(ValueError, match='1'):
            sync_feature.evaluate_batch(inputs(list(expected_evaluate.keys())), conn)
    else:
        assert sync_feature.evaluate_batch(inputs(list(expected_evaluate.keys())), conn).equals(
            pl.Series(name=feature.name,
                      values=[v for _k, v in pairs_with_missing_errors],
                      dtype=feature.raw_dtype.polars_type),
            check_names=True, check_dtypes=True), 'evaluate_batch (substituting None for error outputs)'

    assert (sync_feature.evaluate_batch_safe(inputs([k for k, _v in non_error_pairs]), conn)).equals(
        pl.Series(name=sync_feature.name,
                  values=[expected_safe[k] for k, _v in non_error_pairs],
                  dtype=sync_feature.dtype.polars_type),
        check_names=True, check_dtypes=True), 'evaluate_batch_safe (non error outputs)'

    if single_error_fails_whole_batch:
        assert sync_feature.evaluate_batch_safe(inputs(list(expected_evaluate.keys())), conn).equals(
            pl.Series(name=sync_feature.name,
                      values=[None] * len(expected_evaluate),
                      dtype=feature.dtype.polars_type),
            check_names=True, check_dtypes=True), 'A single error fails the whole batch in evaluate_batch_safe'
    else:
        assert sync_feature.evaluate_batch_safe(inputs(list(expected_evaluate.keys())), conn).equals(
            pl.Series(name=sync_feature.name,
                      values=[v for _k, v in pairs_with_safe_substitutes],
                      dtype=sync_feature.dtype.polars_type),
            check_names=True, check_dtypes=True), 'Errors cause single-row missing values in evaluate_batch_safe'

    assert (sync_feature.evaluate_batch_with_defaults(inputs([k for k, _v in non_error_pairs]), conn)).equals(
        pl.Series(name=sync_feature.name,
                  values=[expected_with_defaults[k] for k, _v in non_error_pairs],
                  dtype=sync_feature.dtype.polars_type),
        check_names=True, check_dtypes=True), 'evaluate_batch_with_defaults (non error outputs)'

    if single_error_fails_whole_batch:
        assert sync_feature.evaluate_batch_with_defaults(inputs(list(expected_evaluate.keys())), conn).equals(
            pl.Series(name=sync_feature.name,
                      values=[substitution_for_none] * len(expected_evaluate),
                      dtype=sync_feature.dtype.polars_type),
            check_names=True, check_dtypes=True), 'A single error fails the whole batch in evaluate_batch_with_defaults (and substitutes the default value)'
    else:
        assert sync_feature.evaluate_batch_with_defaults(inputs(list(expected_evaluate.keys())), conn).equals(
            pl.Series(name=sync_feature.name,
                      values=[substitution_for_none if v is None else v for _k, v in pairs_with_defaults],
                      dtype=sync_feature.dtype.polars_type),
            check_names=True, check_dtypes=True), 'Errors cause single-row missing values in evaluate_batch (and the default value is substituted)'



async def test_int_feature(conn: DuckDBPyConnection) -> None:
    feature = IntTestFeature()
    sync_feature = SyncIntTestFeature()

    expected_evaluate: dict[int | None, int | None | type[Exception]] = { None: -1, 0: None, 1: ValueError }
    safe_substitutions: dict[int | None, int | None] = {}
    defaults_substitutions: list[tuple[int | None, int]] = [(None, feature.default_for_missing)]

    await do_test_feature(feature, sync_feature, conn,
                          expected_evaluate=expected_evaluate,
                          safe_substitutions=safe_substitutions,
                          defaults_substitutions=defaults_substitutions
                          )

    await do_test_feature(BatchIntTestFeature(), BatchSyncIntTestFeature(), conn,
                          expected_evaluate=expected_evaluate,
                          safe_substitutions=safe_substitutions,
                          defaults_substitutions=defaults_substitutions,
                          single_error_fails_whole_batch=True
                          )


async def test_bool_feature(conn: DuckDBPyConnection) -> None:
    feature = BoolTestFeature()
    sync_feature = SyncBoolTestFeature()

    expected_evaluate: dict[int | None, bool | None | type[Exception]] = { None: False, 1: None, 2: True, 3: ValueError }
    safe_substitutions: dict[bool | None, bool | None] = {}
    defaults_substitutions: list[tuple[bool | None, bool]] = [(None, feature.default_for_missing)]

    await do_test_feature(feature, sync_feature, conn,
                      expected_evaluate=expected_evaluate,
                      safe_substitutions=safe_substitutions,
                      defaults_substitutions=defaults_substitutions
                      )

    await do_test_feature(BatchBoolTestFeature(), BatchSyncBoolTestFeature(), conn,
                          expected_evaluate=expected_evaluate,
                          safe_substitutions=safe_substitutions,
                          defaults_substitutions=defaults_substitutions,
                          single_error_fails_whole_batch=True
                          )

async def test_float_feature(conn: DuckDBPyConnection) -> None:
    feature = FloatTestFeature()
    sync_feature = SyncFloatTestFeature()
    
    expected_evaluate: dict[int | None, float | None | type[Exception]] = { None: -1.0, 0: None, 1: ValueError, 2: math.nan, 3: math.inf, 4: -math.inf }
    safe_substitutions: dict[float | None, float | None] = {}
    defaults_substitutions: list[tuple[float | None, float]] = [(None, feature.default_for_missing), (math.nan, feature.default_for_nan),
                                                                (math.inf, feature.default_for_infinity), (-math.inf, feature.default_for_neg_infinity)]
    
    await do_test_feature(feature, sync_feature, conn,
                          expected_evaluate=expected_evaluate,
                          safe_substitutions=safe_substitutions,
                          defaults_substitutions=defaults_substitutions
                          )

    await do_test_feature(BatchFloatTestFeature(), BatchSyncFloatTestFeature(), conn,
                          expected_evaluate=expected_evaluate,
                          safe_substitutions=safe_substitutions,
                          defaults_substitutions=defaults_substitutions,
                          single_error_fails_whole_batch=True
                          )

async def test_categorical_feature(conn: DuckDBPyConnection) -> None:
    feature = CategoricalTestFeature()
    sync_feature = SyncCategoricalTestFeature()

    expected_evaluate: dict[int | None, str | None | type[Exception]] = { None: 'a', 0: None, 1: ValueError, 2: 'b', 3: 'c',
                                                                          4: CategoricalFeature.other_category, 5: 'd', 6: '' }
    safe_substitutions: dict[str | None, str | None] = { 'd': CategoricalFeature.other_category, '': None }
    defaults_substitutions: list[tuple[str | None, str]] = [(None, feature.default_for_missing)]

    await do_test_feature(feature, sync_feature, conn,
                          expected_evaluate=expected_evaluate,
                          safe_substitutions=safe_substitutions,
                          defaults_substitutions=defaults_substitutions
                          )

    await do_test_feature(BatchCategoricalTestFeature(), BatchSyncCategoricalTestFeature(), conn,
                          expected_evaluate=expected_evaluate,
                          safe_substitutions=safe_substitutions,
                          defaults_substitutions=defaults_substitutions,
                          single_error_fails_whole_batch=True
                          )
