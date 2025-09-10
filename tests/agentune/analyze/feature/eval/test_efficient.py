import asyncio
import datetime
import logging
import random
from collections.abc import Generator, Sequence
from functools import reduce
from typing import Any, cast, override

import attrs
import duckdb
import numpy as np
import polars as pl
import pytest
from attrs import define, field, frozen
from duckdb import DuckDBPyConnection
from tests.agentune.analyze.run.feature_search.toys import ToyAsyncFeature

from agentune.analyze.core.dataset import Dataset
from agentune.analyze.feature.eval.base import (
    EfficientEvaluatorParams,
    EfficientEvaluatorProgressCallback,
    EfficientEvaluatorResult,
    FeatureInputs,
    FeatureVariant,
    FeatureVariantEvalState,
)
from agentune.analyze.feature.eval.efficient import (
    RegressionCorrelationMetric,
    SimpleEfficientEvaluator,
)

from .incremental_metrics import generate_regression_data

_logger = logging.getLogger(__name__)

@frozen
class DelayingAsyncFeature(ToyAsyncFeature):
    delay_per_row: datetime.timedelta
    add_random_noise: float = 0.0
    _rnd: random.Random = field(init=False, eq=False, hash=False, factory=lambda: random.Random(42))

    # Redeclare attributes with defaults
    default_for_missing: float = 0.0
    default_for_nan: float = 0.0
    default_for_infinity: float = 0.0
    default_for_neg_infinity: float = 0.0

    @override 
    async def aevaluate(self, args: tuple[Any, ...], 
                        conn: DuckDBPyConnection) -> float:
        await asyncio.sleep(self.delay_per_row.total_seconds())
        value = await super().aevaluate(args, conn)
        noise = self._rnd.normalvariate() * self.add_random_noise
        return value + noise 

    @override
    async def aevaluate_batch(self, input: Dataset, 
                              conn: DuckDBPyConnection) -> pl.Series: 
        await asyncio.sleep(len(input) * self.delay_per_row.total_seconds())
        values = await super().aevaluate_batch(input, conn)
        noises = pl.Series([self._rnd.normalvariate() * self.add_random_noise for _ in range(len(input))])
        return values + noises
    
@define
class CollectingCallback(EfficientEvaluatorProgressCallback):
    should_log: bool = field(default=False)
    initial_states: list[FeatureVariantEvalState] = field(init=False, factory=list)
    updates: list[FeatureVariantEvalState] = field(init=False, factory=list)
    final_states: dict[FeatureVariant, FeatureVariantEvalState] = field(init=False, factory=dict)

    @override
    async def starting_states(self, states: Sequence[FeatureVariantEvalState]) -> None: 
       self.initial_states = list(states)
       self.final_states = { s.variant: s for s in states }
       if self.should_log:
           for state in states:
               _logger.info(f'''Starting {state.variant} with metric={state.metric_value} 
                                and uncertainty={state.metric_uncertainty}''')

    @override
    async def update(self, state: FeatureVariantEvalState) -> None: 
        assert state != self.final_states[state.variant], f'Variant {state.variant} already in state {state}'
        self.updates.append(state)
        self.final_states[state.variant] = state
        if self.should_log:
            if state.discarded_reason:
                _logger.info(f'''Discarded {state.variant} with reason={state.discarded_reason}, {len(state.feature_outputs)} rows, 
                            metric={state.metric_value} and uncertainty={state.metric_uncertainty}''')
            else:
                _logger.info(f'''Updated {state.variant} with {len(state.feature_outputs)} rows, metric={state.metric_value},
                            uncertainty={state.metric_uncertainty}, effective_cost_per_row={state.effective_cost_per_row}''')

@frozen
class NamedVariant(FeatureVariant):
    name: str = field()
    @name.default # Only needed because we can't add a mandatory field after the superclass's fields that have default values
    def _default_name(self) -> str:
        return f'{self.feature.name}-{self.cost_per_row}'

    @override
    def __str__(self) -> str:
        return self.name    
    
    @override
    def __repr__(self) -> str:
        return self.name    

@pytest.fixture(scope='module')
def duckdb_conn() -> Generator[DuckDBPyConnection, None, None]:
    with duckdb.connect(f':memory:{__name__}') as conn:
        yield conn

@pytest.fixture(scope='module')
def true_correlation() -> float:
    return 0.3

@pytest.fixture(scope='module')
def regression_data(true_correlation: float) -> pl.DataFrame:
    return generate_regression_data(10_000, true_correlation, 42)

@pytest.fixture(scope='module')
def default_params(true_correlation: float) -> EfficientEvaluatorParams:
    return EfficientEvaluatorParams(min_metric=true_correlation * 0.8, 
                                    max_metric_uncertainty=0.05, metric_per_cost_min_roi=0.05, 
                                    evaluate_chosen_variant_on_all_data=False)

def make_variant(name: str, cost: float | None, noise: float = 0.0, delay: datetime.timedelta = datetime.timedelta(0)) -> NamedVariant:
    return NamedVariant(DelayingAsyncFeature('a', 'b', f'cost={cost}', '', '', delay, noise), cost_per_row=cost, name=name)

async def choose_assert(duckdb_conn: DuckDBPyConnection,
                        regression_data: pl.DataFrame,
                        variants: Sequence[FeatureVariant], 
                        expected_choice: FeatureVariant | None, 
                        params: EfficientEvaluatorParams,
                        callback: CollectingCallback | None = None,
                        should_log: bool = False) -> EfficientEvaluatorResult:
    data = regression_data.with_columns(a=regression_data['feature'] * 0.25, b=regression_data['feature'] * 0.75) \
                          .drop('feature')

    inputs = FeatureInputs(
        Dataset.from_polars(data),
        'target',
        duckdb_conn
    )
    metric = RegressionCorrelationMetric()
    
    if callback is None:
        callback = CollectingCallback(should_log=should_log)
    result = await SimpleEfficientEvaluator().choose(variants, inputs, metric, params, callback)

    if params.max_cost_spent_evaluating is not None:
        assert result.total_cost_spent <= params.max_cost_spent_evaluating, 'Did not spend more than allowed cost evaluating'
    if params.max_time_spent_evaluating is not None:
        assert result.total_time_spent <= params.max_time_spent_evaluating, 'Did not spend more than allowed time evaluating'
    
    assert { s.variant for s in callback.initial_states } == set(variants), 'Reported initial state for all variants'
    assert callback.final_states.keys() == set(variants), 'Reported final state for all variants'
    for state in callback.final_states.values():
        if state != result.chosen_variant:
            assert state.discarded_reason is not None, f'Non-chosen variant {state.variant} must have reason for being discarded'
    
    # TODO can also test callback.updates to check it matches the transition from initial to final state

    assert result.total_cost_spent == sum(state.cost_spent for state in callback.final_states.values()), \
        'Total cost spent is the sum of the final states'
    assert result.total_time_spent == reduce(lambda x, y: x+y, [state.time_spent for state in callback.final_states.values()], datetime.timedelta(0)), \
        'Total time spent is the sum of the final states'

    chosen_variant = result.chosen_variant
    if chosen_variant is not None:
        assert chosen_variant.variant == expected_choice, f'Chosen variant is {chosen_variant.variant} but expected {expected_choice}'
        assert chosen_variant.metric_value >= params.min_metric 
        assert chosen_variant.metric_uncertainty <= params.max_metric_uncertainty
    else:
        assert expected_choice is None, f'Expected {expected_choice} but got no choice'

    return result

async def test_single_variant(duckdb_conn: DuckDBPyConnection,
                              regression_data: pl.DataFrame,
                              default_params: EfficientEvaluatorParams) -> None:
    variants = [make_variant('one', 1.0)]
    expected_choice = variants[0]
    await choose_assert(duckdb_conn, regression_data, variants, expected_choice, default_params)

async def test_same_signal_different_cost(duckdb_conn: DuckDBPyConnection,
                              regression_data: pl.DataFrame,
                              default_params: EfficientEvaluatorParams) -> None:
    variants = [make_variant(str(i), i) for i in [2.0, 1.0, 3.0]]
    expected_choice = variants[1] # Cheaper is better
    await choose_assert(duckdb_conn, regression_data, variants, expected_choice, default_params)

async def test_not_enough_budget_for_signal(duckdb_conn: DuckDBPyConnection,
                              regression_data: pl.DataFrame,
                              default_params: EfficientEvaluatorParams) -> None:
    low_eval_cost_params = attrs.evolve(default_params, max_cost_spent_evaluating=100.0)
    variants = [make_variant(str(i), i) for i in [2.0, 1.0, 3.0]]
    expected_choice = None
    await choose_assert(duckdb_conn, regression_data, variants, expected_choice, low_eval_cost_params)


async def test_same_cost_different_signal(duckdb_conn: DuckDBPyConnection,
                              regression_data: pl.DataFrame,
                              default_params: EfficientEvaluatorParams) -> None:
    # Low ROI threshold, choose most expensive but most effective variant
    noisy_variants = [make_variant(str(cost), cost=cost, noise=0.15 * (10-cost)) for cost in range(0, 8, 1)]
    params = attrs.evolve(default_params, metric_per_cost_min_roi=0.001)
    expected_variant = noisy_variants[-1]
    await choose_assert(duckdb_conn, regression_data, noisy_variants, expected_variant, params)

    # With higher ROI threshold, choose a middling variant
    # Reinit the per-feature random generators or things go wrong
    noisy_variants = [make_variant(str(cost), cost=cost, noise=0.15 * (10-cost)) for cost in range(0, 8, 1)]
    params = attrs.evolve(default_params, metric_per_cost_min_roi=1.0)
    expected_variant = noisy_variants[4]
    await choose_assert(duckdb_conn, regression_data, noisy_variants, expected_variant, params)

async def test_all_variants_low_signal(duckdb_conn: DuckDBPyConnection,
                              regression_data: pl.DataFrame,
                              default_params: EfficientEvaluatorParams) -> None:
    # All variants have signal too low to choose
    variants = [make_variant(str(noise), cost=1.0, noise=cast(float, noise)) for noise in np.arange(0.5, 0.9, 0.1)]
    expected_variant = None
    params = attrs.evolve(default_params, min_metric=0.5)
    await choose_assert(duckdb_conn, regression_data, variants, expected_variant, params)

async def test_time_cost_conversion(duckdb_conn: DuckDBPyConnection,
                              regression_data: pl.DataFrame,
                              default_params: EfficientEvaluatorParams) -> None:
    feature = DelayingAsyncFeature('a', 'b', 'a+b', '', '', datetime.timedelta(0),
                                   default_for_nan=0.0, default_for_missing=0.0,
                                   default_for_infinity=0.0, default_for_neg_infinity=0.0)
    variant_cost = NamedVariant(feature, cost_per_row=1.0, name='cost=1.0')
    variant_time = NamedVariant(feature, time_per_row=datetime.timedelta(seconds=2.0), name='time=2.0')
    variant_both = NamedVariant(feature, cost_per_row=5.0, time_per_row=datetime.timedelta(seconds=7.0), name='cost=5.0, time=7.0')
    variants = [variant_cost, variant_time, variant_both]
    expected_choice = variant_cost
    callback = CollectingCallback()
    await choose_assert(duckdb_conn, regression_data, variants, expected_choice, default_params, callback=callback)
    
    assert callback.final_states[variant_cost].effective_cost_per_row == variant_cost.cost_per_row
    assert callback.final_states[variant_both].effective_cost_per_row == variant_both.cost_per_row
    assert callback.final_states[variant_time].effective_cost_per_row == cast(datetime.timedelta, variant_time.time_per_row).total_seconds() * default_params.cost_per_second

async def test_time_measurement(duckdb_conn: DuckDBPyConnection,
                              regression_data: pl.DataFrame,
                              default_params: EfficientEvaluatorParams) -> None:
    variants = [make_variant(str(i), cost=None, delay = datetime.timedelta(milliseconds=i)) for i in range(1, 5, 1)]

    expected_choice = variants[0] # Least delay is best when all have same cost
    callback = CollectingCallback()
    await choose_assert(duckdb_conn, regression_data, variants, expected_choice, default_params, callback=callback)

    for variant in variants:
        state = callback.final_states[variant]
        feature = cast(DelayingAsyncFeature, variant.feature)

        actual_time = state.time_spent.total_seconds()
        expected_time = (len(state.feature_outputs) * feature.delay_per_row).total_seconds()
        assert actual_time == pytest.approx(expected_time, rel=0.1), f'Time spent is {actual_time} but expected {expected_time}'
        
        expected_cost = default_params.cost_per_second * state.time_spent.total_seconds()
        actual_cost = state.effective_cost_per_row
        assert actual_cost == pytest.approx(expected_cost, rel=0.05), f'Cost is {actual_cost} but expected {expected_cost}'

