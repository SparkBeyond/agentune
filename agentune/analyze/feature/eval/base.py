import asyncio
import datetime
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self, override

import polars as pl
from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.core.dataset import Dataset
from agentune.analyze.feature.base import Feature
from agentune.analyze.progress.base import ProgressStage
from agentune.analyze.util.polarutil import series_field


class FeatureEvaluator(ABC):
    """A feature evaluator can evaluate many features at once more efficiently than calling each feature's evaluate method one by one.
    
    This works only for features with particular similiarities: e.g. a group of async features, SQL query features, or AST-based features.
    """

    @classmethod
    @abstractmethod
    def supports_feature(cls, feature: Feature) -> bool:
        """Whether this evaluator can evaluate this feature, together with other features for which it returns True,
        more efficiently than evaluating them one by one (or in parallel in the case of async features).
        """
        raise NotImplementedError # returning ... evaluates as false in a boolean context

    @classmethod
    @abstractmethod
    def for_features(cls, features: Sequence[Feature]) -> Self: ...

    @property
    @abstractmethod
    def features(self) -> Sequence[Feature]:
        ...

    @abstractmethod
    async def aevaluate(self, dataset: Dataset, conn: DuckDBPyConnection,
                        cells_progress: ProgressStage | None = None) -> Dataset:
        """Args:
            dataset: includes all columns needed by all the features. Any additional columns must be ignored by the implementation.
            conn: makes available contains data declared in `secondary_tables` or `join_strategies`.
                  Any additional tables or columns must be ignored by the implementation.
            cells_progress: will be used to increment the count of cells (i.e. rows*features) evaluated.
                            If not given, a new stage will be created for the duration of the call.

        Returns:
            A dataset with a column per feature, named with the feature's name.
        """
        ...

class SyncFeatureEvaluator(FeatureEvaluator):
    @abstractmethod
    def evaluate(self, dataset: Dataset, conn: DuckDBPyConnection,
                 cells_progress: ProgressStage | None = None) -> Dataset:
        """See FeatureEvaluator.aevaluate for details."""
        ...

    @override
    async def aevaluate(self, dataset: Dataset, conn: DuckDBPyConnection,
                        cells_progress: ProgressStage | None = None) -> Dataset:
        with conn.cursor() as cursor:
            return await asyncio.to_thread(self.evaluate, dataset.copy_to_thread(), cursor, cells_progress)

# The following classes make up the API of EfficientEvaluator (which comes last)

class EfficientEvaluatorMetric(ABC):
    @abstractmethod
    def calculate(self, outputs: pl.Series, target: pl.Series) -> tuple[float, float]:
        """Returns (metric, uncertainty)"""
        ...

@frozen
class FeatureInputs:
   """The parameters to Feature.evaluate, plus a target column."""

   input: Dataset
   target_column_name: str
   conn: DuckDBPyConnection

   @property
   def target_column(self) -> pl.Series:
      return self.input.data.get_column(self.target_column_name)


@frozen
class FeatureVariant:
   # We always call batch_evaluate() not evaluate(), even if batch_size is 1.
   # This guarantee makes it easier to reason about the performance of a given implmenetation (variant).
   feature: Feature
   
   # Either the `cost` or `time` is used if known (with a linear conversion factor between the two
   # given below); if both are unknown, the time is empirically measured and used.
   cost_per_row: float | None = None # Cost per row, in abstract money units
   time_per_row: datetime.timedelta | None = None # Time per row. If unknown, may be measured empirically.
   min_rows_to_estimate_time_cost: float = 20 # May be batched

   # May make calls in parallel with other variants that also have this flag set, even when one or both are measuring performance.
   # We will still not make parallel calls to the same variant; rely on batch_size instead.
   may_parallelize_with_others: bool = False

   batch_size: int | None = None # Batch size to use for optimal cost / performance.
                                    # If given, overrides native logic, even when this would result in
                                    # spending more time or cost evaluating a variant than otherwise needed.
                                    # If not given, we use the default_batch_size.

@frozen
class FeatureVariantEvalState:
   """API representation of EfficientEvaluator intermediate state, for progress reporting."""

   variant: FeatureVariant
   metric_value: float
   metric_uncertainty: float
   time_spent: datetime.timedelta
   cost_spent: float
   discarded_reason: str | None # If the variant is no longer under consideration
   
   # If the variant has known cost or time, calculated from that.
   # Otherwise, can be measured empirically, and remains None until then.
   effective_cost_per_row: float | None

   feature_outputs: pl.Series = series_field()


class EfficientEvaluatorProgressCallback:
   @abstractmethod
   async def starting_states(self, states: Sequence[FeatureVariantEvalState]) -> None: 
       """Called once at the beginning to inform you of the states we assign to all variants."""
       ...

   @abstractmethod
   async def update(self, state: FeatureVariantEvalState) -> None: 
       """Called whenever a variant's state changes, including when a variant is chosen."""
       ...

@frozen
class EfficientEvaluatorResult:
   chosen_variant: FeatureVariantEvalState | None # If not None, status is Chosen
   # Total time & cost spent evaluating all variants. This is cumulative time in evaluate(), not wall clock time.
   total_time_spent: datetime.timedelta 
   total_cost_spent: float

@frozen
class EfficientEvaluatorParams:
    """Parameters influcencing the choice. Defined as a separate class for convenience."""

    min_metric: float # Chosen variant must have metric >= this
    max_metric_uncertainty: float # Max uncertainty allowed before taking calculated metric into account. E.g. 0.05

    # Investing 1.0 more cost must provide at least metric_per_cost_min_roi better metric for us to choose a more expensive variant.
    # (This is tied to cost because cost and time are related by a constant factor.)
    # The value used will be different depending on the metric.
    metric_per_cost_min_roi: float

    # TODO the above is not transitive; it seems we should instead choose the global optimum?

    cost_per_second: float = 1.0 # Tradeoff factor between cost and time.
    max_cost: float | None = None # Chosen variant must have cost <= this
    max_time: datetime.timedelta | None = None # Chosen variant must have time <= this
    default_batch_size: int = 20 # If set to 1, we call the per-row evaluate() not batch_evaluate()

    # Max time & cost spent on evaluating different variants before choosing one.
    # If we can then choose a variant satisfying the other constraints, we will still evaluate the rest of the rows
    # even if that exceeds this budget.
    max_cost_spent_evaluating: float | None = None
    max_time_spent_evaluating: datetime.timedelta | None = None # Cumulative variant-call time in case of parallel evaluation

    # Once we've chosen a variant, whether to evaluate it on all input data or to return immediately
    evaluate_chosen_variant_on_all_data: bool = True


class EfficientEvaluator(ABC):
    """Chooses one of several variants of the same feature, based on a cost/metric tradeoff.
    
    Implementations can have more parameters at the class level to control their behavior.
    """

    @abstractmethod
    async def choose(self, variants: Sequence[FeatureVariant], inputs: FeatureInputs, metric: EfficientEvaluatorMetric,
                     params: EfficientEvaluatorParams, progress_callback: EfficientEvaluatorProgressCallback) -> EfficientEvaluatorResult: 
        """Returns the cheapest variant s.t. choosing the next more expensive one instead would not be worth the ROI.
        If no such variant exists, or no variant satisfies the other constraints, returns None.
        """
        ...
