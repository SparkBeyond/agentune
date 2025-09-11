from agentune.analyze.join.base import JoinStrategy

# Features (for developers)

## What is a feature?

A feature - an instance of class Feature - is defined as: a function that calculates a value of a known type for a row of input data.

There are other useful concepts and entities, which are not functions or are not per row or don't have a useful type.
They are not called 'features' in this codebase. Talking about 'this is not what features should be like' is not productive;
this is the definition of "feature".

## Feature types

There are four feature types currently defined: int, float, bool, and categorical. Int features return signed int32 values.
Float features return float64 values (including nan and +/- inf).

Categorical features return string values out of a known list (given by `feature.categories`) and an extra, special value
given by `CategoricalFeature.other_category`. This is returned if there is an unexpected value or if there was a long tail
of rare values that did not make it into the categories list. The special value is not itself included in the categories list.

There are no 'open ended' string features right now.

All features can return missing values (represented by None in scalars).

The type of a Feature instance can be determined by checking isinstance against the dedicated subclass: IntFeature, FloatFeature,
BoolFeature, and CategoricalFeature; or by checking `feature.dtype` (this is less recommended). 

## Feature.evaluate signature

There are two core methods implemented by Feature: row (scalar) evaluate and batch evaluate.
They should not be called by users directly; the rest of this document describes progressively 
higher-level APIs. 

```python
async def aevaluate(self, args: tuple[Any, ...], 
                    conn: DuckDBPyConnection) -> T | None:
async def aevaluate_batch(self, input: Dataset, 
                          conn: DuckDBPyConnection) -> pl.Series:
```

A feature typically overrides only one of these two methods; the default implementation of each 
calls the other one. (The default implementation of `aevaluate_batch` may change in the future
to enable better control of concurrent evaluation.)

The main dataset is given either by `args` (for a single row) or by `input` (for a batch of rows).
A future update will add support for implementing aevaluate by declaring the specific arguments used,
e.g. `async def aevaluate(self, id: int, name: str, conn: ...)` (#144).

The secondary datasets are available in duckdb using `conn`. A feature can store any `JoinStrategy` instances
that were available during feature generation and use them to query the data, or it can create one later;
they do not represent the availability of additional data.

The main table can also be made available via `conn` to run queries that join it to secondary tables,
by using `conn.register`; see `SqlFeature.evaluate_batch` for an example. 

All data, both input and output, can contain missing values (`na` in Polars, `null` in SQL).

When implementing asynchronous APIs that deal with data, keep in mind the [threading rules](threading.md).
Operations in duckdb and in polars are always synchronous.

## Feature metadata

### Feature.name

This is the name given to the feature's outputs as a column in enriched datasets (including DB tables).
All strings are currently valid, although of course feature generators are encouraged to use user-friendly names.

If several features being enriched in the same dataset have the same name, the outupt names will be deduplicated
by adding one or more underscores as a suffix; see `dedup_names.py`.

### Feature.description and Feature.technical_description

Both of these are meant for humans to read. The first describes what the feature tries to do, ideally in simple language,
which can be imprecise; the second describes how it does it in more detail, possibly with code or pseudocode.

For example, a feature's description might say "is male" and the technical description "name contains 'mr.'". 

The design discussion is ongoing in #62.

### Feature schema metadata

A Feature instance has three properties describing the input data it requires:

```python
def params(self) -> Schema
def secondary_tables(self) -> Sequence[DuckdbTable]
def join_strategies(self) -> Sequence[JoinStrategy]
```

These are subsets of the data that was available during feature search when the feature was created.
Specifying what the feature uses lets users run Enrich on new data without providing columns or tables
that no selected features use.

`secondary_tables` should be used if the feature runs SQL queries on them; `join_strategies` should be used
if it uses those objects to access data. Both can be specified.

When `Feature.evaluate` variants are called, only the data declared in `params` is actually passed in the `args`
parameter. The `args` are passed in the order in which they were declared in `params`, which may be different
from the order of these columns in the feature search input dataset. Similarly, the dataset passed to `evaluate_batch`
includes only the columns declared in `params`. The data available through the DuckDB connection includes only the 
`secondary_tables` (and only the columns listed for them in `secondary_tables`), as well as any tables and columns
referenced by the declared `join_strategies`.

More data MAY be available through the DuckDB connection and in additional columns of the main dataset passed to 
`evaluate_batch`, but the feature must not access it or rely on it.

## Safe feature evaluation (evaluate_safe)

The implementation of `evaluate` SHOULD always return a value of the correct type. But, because we don't implement
all features ourselves, we need to guard against bugs and incorrect behavior. This is done in the wrapper methods:

```python
async def aevaluate_safe(self, args: tuple[Any, ...], 
                         conn: DuckDBPyConnection) -> T | None:
async def aevaluate_batch_safe(self, input: Dataset, 
                               conn: DuckDBPyConnection) -> pl.Series:
```

These wrap the base `aevaluate` and `aevaluate_batch` methods, catch all errors and return missing values.
For categorical features, they also replace return values that are not in the declared categories list with 
`CategoricalFeature.other_category`, and replace the empty string with a missing value.

All higher-level code (e.g. EnrichRunner) uses the _safe wrapper methods. Features must not override them.

Note that, if `aevaluate_batch` raises an error, the output for entire batch becomes missing values.
This may be changed in the future (#155).

## Feature evaluation with default values

Each Feature instance stores a default value which can be used instead of the missing value in outputs.
This is defined as an attribute `default_for_missing: T`, so you can use `attrs.evolve` to change it.

Float features have three additional default values which can be substituted for non-finite values: 
`default_for_nan`, `default_for_inf`, and `default_for_neg_inf`.

You can substitute these values into the output of `evaluate_safe` by calling these Feature methods:

```python
def substitute_defaults(self, value: T | None) -> T:
def subsistute_defaults_batch(self, values: pl.Series) -> pl.Series:
```

(These methods are synchronous because they admit no other / asynchronous implementation; they are very fast in practice.)

You can also call these convenience methods, which combine `evaluate_safe` and `substitute_defaults`:

```python
async def aevaluate_with_defaults(self, args: tuple[Any, ...], 
                                  conn: DuckDBPyConnection) -> T:
async def aevaluate_batch_with_defaults(self, input: Dataset, 
                                        conn: DuckDBPyConnection) -> pl.Series:
```

Note that `aevaluate_with_defaults` returns `T` and not `T | None`.

Each component needs to decide whether to support missing values and/or non-finite float values.
The enriched data passed around is (normally) the output of `evaluate_safe`, with missing and non-finite values,
because substituting the defaults is very cheap and can be done whenever it's needed. High-level user APIs
such as EnrichRunner may, in the future, add support for returning the output with defaults, or both sets of outputs.

### FeatureGenerator and default values

When a `FeatureGenerator` generates a feature, it returns a wrapper class `GeneratedFeature`, which contains an attribute
`has_good_defaults: bool`. If it is True, the default value attributes on the feature are not changed. 

If it is False the feature search substitutes 'default default' values, which are calculated on the feature's outputs 
on the feature search dataset, as follows:

1. For Bool features, False
2. For Categorical features, CategoricalFeature.other_category
3. For int features, the median. It can be a value that does not itself appear in the feature's output.
4. For float features:
    1. The defaults for infinity and -infinity are the max finite value +1 and the min finite value -1, respectively.
    2. The default for both missing values and `nan` is the median, calculated after substituting defaults for infinity and -infinity,
       and ignoring `nan` values.

## Synchronous features

A feature's implementation may be asynchronous, as shown above, or synchronous. Feature implementations SHOULD
be synchronous if they don't await anything.

Synchronous features implement the subclass `SyncFeature` and the appropriate per-type subclass (`SyncIntFeature`, etc.).
The evaluate method has the same signature as before, apart from being synchronous and being called `evaluate` not `aevaluate`:

```python
def evaluate(self, args: tuple[Any, ...], 
             conn: DuckDBPyConnection) -> T | None:
```

There are synchronous equivalents of all the other methods mentioned above: _batch, _safe and _with_defaults variants.

Although SyncFeature extends Feature, its asynchronous methods SHOULD NOT be called. All code handling features has separate
codepaths for synchronous and asynchronous features.

## Efficient evaluation strategies (FeatureEvaluator)

Sometimes it's possible to evaluate several features on the same data more efficiently than calling them one by one
(or in parallel, in the case of asynchronous features). Each such strategy is specific to some particular Feature implementation.
These strategies are implemented by FeatureEvaluator subclasses. In Enrich, each available evaluator can choose which features
to operate on.

There are no current implementations of FeatureEvaluator (other than the trivial ones used by default for all features).
Implementations we may add in the future include:
- LLM-backed features: we may want to evaluate one feature first, to insert it into the provider's token cache,
  and then evaluate all other features on the same row in parallel
- Function-composition features: we may want to evaluate common subfunctions only once on the same inputs,
  both across features and across rows

## Implementing new features

Extend the correct subclass of Feature: feature type X synchronicity, e.g. float + sync = SyncFloatFeature.
(If you need to generate features of different types, you will need to implement a subclass for each type.)

Extend any additional subclasses of Feature that describe your feature:
- LlmFeature, for features accessing an LLM at runtime
- SqlQueryFeature or SqlBackedFeature, for features defined by a single SQL query
- WrappedFeature, if you wrap another feature instance and modify its output

Override either `evaluate` or `evaluate_batch` or both: 
- Override `evaluate_batch` if it is a real batch implementation; don't restate the default implementation which loops over rows. However, it may be appropriate to change the default behavior to e.g. change the degree of parallelism in `aevaluate_batch`.
- If you override `evaluate_batch`, check if you can override `evaluate` to be cheaper than the default implementation (which wraps each row in a Dataset to call evaluate_batch). All code calls `evaluate_batch` when possible in preference to `evaluate`, but some (e.g. user) code may still call `evaluate`.

Decorate your class with `@attrs.define` (or, preferrably, `@attrs.frozen`).

Implement the metadata properties; use attributes if possible (i.e. they don't need methods to compute them).

Make sure serialization works.

## Serializing features to json

TBD (to be done and to be documented); see #157.

## Evaluating features (as a user)

The high-level API that users, and if possible all code components, use to evaluate one or more features is `EnrichRunner`. 
It adds support for things individual features (and FeatureEvaluators) don't need to handle, e.g. feature name deduplication, 
keeping some input columns in the output and choosing whether to substitute feature defaults in the output.

When you have a Feature instance in hand it's tempting to call one of its evaluate methods directly. You should be aware of
the functionality added by EnrichRunner that you would be missing out on.
