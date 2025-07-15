import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Literal, override

import polars as pl
from duckdb import DuckDBPyConnection
from llama_index.core.llms import ChatMessage, ChatResponse

import agentune.analyze.core.types
from agentune.analyze.context.base import ContextDefinition, TablesWithContextDefinitions
from agentune.analyze.core.database import DatabaseTable
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Schema
from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.core.types import Dtype

type TargetKind = Literal['classification', 'regression']
type Classification = Literal['classification']
type Regression = Literal['regression']


class Feature[T](ABC):
    """Args:
        name: Used as the column/series name in outputs. Not guaranteed to be unique among Feature instances.
        description: Human-readable description of the feature.
        code: Python code that computes this feature. The code should define a function `evaluate` with the needed params. 
              SQL features will expose code that executes the SQL query.
              
    Type parameters:
        T: The type of the feature's output values, when they appear as scalars.
           This is not a free type parameter; only the values defined by the subclasses below, such as IntFeature, are allowed.
           Note that features with different dtypes can have the same scalar T, e.g. features with dtype Int32 and Int64 would
           both have T=int. (There is no feature type using Int64 at the moment of writing, but you should not write code
           that assumes all features have distinct T types.)
    """

    @property
    @abstractmethod
    def dtype(self) -> Dtype: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    # TODO precise definition - 
    #  We have not yet decided whether the code needs to be strictly correct and/or guaranteed to evaluate and work.
    @property
    @abstractmethod
    def code(self) -> str: ...

    @property
    @abstractmethod
    def params(self) -> Schema: 
        """Columns of the main table used by the feature.
        This affects the parameters to evaluate().
        """
        ...
    
    @property
    @abstractmethod
    def context_tables(self) -> Iterable[DatabaseTable]: 
        """Context tables used by the feature.
        This affects the parameters to evaluate().
        Specifying a table with only the columns you will use may, in future, allow us
        to improve performance.
        """
        ...

    @property
    @abstractmethod
    def context_objects(self) -> Iterable[ContextDefinition]: 
        """Context object definitions used by the feature.
        This affects the parameters to evaluate().
        Specifying a context with only the value columns you will use may, in future, allow us
        to improve performance.
        """
        ...

    @abstractmethod
    def is_numeric(self) -> bool: ...

    # A feature must override at least one of evaluate or evaluate_batch.

    async def aevaluate(self, args: tuple[Any, ...], contexts: TablesWithContextDefinitions,
                        conn: DuckDBPyConnection) -> T: 
        """Evaluate a single row.

        The arguments `args` are in the order given by `self.params`.

        The default implementation delegates to evaluate_batch and is quite inefficient;
        if you override the batch implementation, please consider if you can also override this one
        more efficiently.

        All context tables are available in the provided `conn`ection.
        """
        df = pl.DataFrame(
            {col.name: [value] for col, value in zip(self.params.cols, args, strict=True)},
            schema=self.params.to_polars()
        )
        return (await self.aevaluate_batch(Dataset(self.params, df), contexts, conn))[0]

    async def aevaluate_batch(self, input: Dataset, contexts: TablesWithContextDefinitions,
                              conn: DuckDBPyConnection) -> pl.Series: 
        strict_df = pl.DataFrame([input.data.get_column(col.name) for col in self.params.cols])
        results = asyncio.gather(*[self.aevaluate(row, contexts, conn) for row in strict_df.iter_rows()])
        return pl.Series(name=self.name, dtype=self.dtype.polars_type, values=results)


# Every feature must implement exactly one of the feature value type interfaces (IntFeature, etc) - 
# it is not enough to directly implement e.g. Feature[int].

# -------- Feature value types

# Note that the values of type param T are all non-None; missing values are allowed in feature outputs.
# Generally speaking, features should only return missing values if one of the inputs has a missing value.
# We might prefer a simpler world where feature outputs can't be missing, but that would not allow us
# to use input columns as features.

class NumericFeature[T](Feature[T]): 
    @override
    def is_numeric(self) -> bool: return True

# Other int sizes or unsigned ints can be added as needed.
class IntFeature(NumericFeature[int | None]): 
    @property
    @override
    def dtype(self) -> Dtype: return agentune.analyze.core.types.int32

class FloatFeature(NumericFeature[float | None]):
    @property
    @override
    def dtype(self) -> Dtype: return agentune.analyze.core.types.float64

class BoolFeature(Feature[bool | None]): 
    @override
    def is_numeric(self) -> bool: return False

    @property
    @override
    def dtype(self) -> Dtype: return agentune.analyze.core.types.boolean

class CategoricalFeature(Feature[str | None]): 
    @property
    @abstractmethod
    def categories(self) -> Iterable[str]: ...
    
    @override
    def is_numeric(self) -> bool: return False

    @property
    @override
    def dtype(self) -> Dtype: return agentune.analyze.core.types.string

# -------- Other feature types

class SyncFeature[T](Feature[T]):
    # A feature must override at least one of evaluate or evaluate_batch.

    def evaluate(self, args: tuple[Any, ...], contexts: TablesWithContextDefinitions,
                 conn: DuckDBPyConnection) -> T: 
        df = pl.DataFrame(
            {col.name: [value] for col, value in zip(self.params.cols, args, strict=True)},
            schema=self.params.to_polars()
        )
        return self.evaluate_batch(Dataset(self.params, df), contexts, conn)[0]

    def evaluate_batch(self, input: Dataset, contexts: TablesWithContextDefinitions,
                       conn: DuckDBPyConnection) -> pl.Series: 
        strict_df = pl.DataFrame([input.data.get_column(col.name) for col in self.params.cols])
        return pl.Series(name=self.name, dtype=self.dtype.polars_type, 
                         values=[self.evaluate(row, contexts, conn) for row in strict_df.iter_rows()])   

    @override 
    async def aevaluate(self, args: tuple[Any, ...], contexts: TablesWithContextDefinitions,
                       conn: DuckDBPyConnection) -> T: 
        # TODO here too, need to adapt the connection to a new thread
        return await asyncio.to_thread(self.evaluate, args, contexts, conn.cursor())

    @override
    async def aevaluate_batch(self, input: Dataset, contexts: TablesWithContextDefinitions,
                              conn: DuckDBPyConnection) -> pl.Series: 
        return await asyncio.to_thread(self.evaluate_batch, input, contexts, conn.cursor())
    

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

class SinglePromptLlmFeature[T](LlmFeature[T]):
    """A feature that is evaluated by an LLM with a single prompt per input row."""

    @abstractmethod 
    def row_prompt(self, args: tuple[Any, ...], contexts: TablesWithContextDefinitions,
                   conn: DuckDBPyConnection) -> Sequence[ChatMessage]: ...

    @abstractmethod
    def parse_result(self, result: ChatResponse) -> T: ...

    @override
    async def aevaluate(self, args: tuple[Any, ...], contexts: TablesWithContextDefinitions,
                        conn: DuckDBPyConnection) -> T: 
        return self.parse_result(await self.model.llm.achat(self.row_prompt(args, contexts, conn)))
