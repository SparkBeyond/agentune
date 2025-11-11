# Examples for using the Agentune API

## Generating more features; passing parameters to components

Usecase: you want `analyze` to return more or fewer features. 

There is a parameter to `analyze` which you can change directly:

```python
ctx.ops.analyze(..., params=AnalyzeParams(max_features_to_select=100))
```

But setting this parameter is not enough; we need to generate enough candidate features to be able to select as many features as we want. To affect candidate features, we need to pass parameters to the feature generator. Because these are parameters specific to a particular feature generator, they are not included in `AnalyzeParams`, and to pass them we need to know (and write in our code) which feature generator to use. 

In this example, the feature generator is `ConversationQueryFeatureGenerator`, which has the parameters `num_features_per_round` and `num_actionable_rounds` (among others). Part of its algorithm generates `num_features_per_round` * `num_actionable_rounds` features, and we want to increase these values:

```python
import attrs

default_generator = ctx.defaults.conversation_query_feature_generator()
custom_generator = attrs.evolve(default_generator, num_features_per_round=30, num_actionable_rounds=3)

default_components = ctx.defaults.analyze_components()
custom_components = attrs.evolve(default_components, generators=(custom_generator, ))

results = await ctx.ops.analyze(..., components=custom_components)
```

There are several things to note here:

1. Default parameter values, including default instances of components such as feature generators, can be obtained from the `ctx.defaults` namespace.
2. Start by acquiring the default value and modify only the parameters you want to set. 
  
    While you could create the entire component instance from scratch (e.g. by calling `ConversationQueryFeatureGenerator(...)`), you would need to correctly construct all the parameters, and you might need to update your code if the default value of another parameter changes in the future. 
3. Most classes in agentune are immutable. To modify a parameter you need to create a new copy, which is done by `attrs.evolve`. On Python 3.13+ you can use the standard library method `copy.copy` instead.
4. The class `AnalyzeComponents` specifies all the components (i.e. classes implementing functional APIs) used by `analyze`, including the feature generator(s).

## Using more examples when recommending actions

Usecase: you want the action recommender to use a bigger sample of the input to generate recommendations.

Similarly to the previous example, this requires customizing a component. In this case the component is `ConversationActionRecommender`, which has a parameter called `max_samples`:

```python
import attrs

default_recommender = ctx.defaults.conversation_action_recommender()
custom_recommender = attrs.evolve(default_recommender, max_samples=100)
```

However, we cannot pass this component directly to the method `ctx.ops.recommend_conversation_actions`. Instead, we have to call a more general method called `recommend_actions`:

```python
recommendations = await ctx.ops.recommend_actions(analyze_input, analyze_results, custom_recommender)
```

## Passing a separate test dataset

Usecase: you want to provide test data from a different source (file, table, etc.) from the train data.

Ingest both datasets into the system:

```python
train = ctx.data.from_csv('train.csv').copy_to_table('train')
test = ctx.data.from_csv('test.csv').copy_to_table('test')
```

Split the train dataset, but (unlike normally) do not assign any rows to the test split:
```python
split_train = train.split(train_fraction=1.0)
```

When the train fraction is 1.0, the test split is empty. (Splitting the input table is still required because it also creates sub-samples of the train split which we use internally.)

Finally, pass both inputs to `analyze`:

```python
results = await ctx.ops.analyze(..., train_input=split_train, test_input=test)
```

Note: when the `test_input` parameter is provided to `analyze`, the test split of the `train_input` is ignored, whether or not it is empty.

## Using different LLM models

Usecase: you want to generate or evaluate features using a non-default LLM.

This requires passing custom parameters to the feature generator, as in the first example, so read that one first.

An LLM model is specified by an instance of class `LLMSpec`. You need to first convert it to a live LLM, and then to pass the result to the component you want to use it:

```python
import attrs
from agentune.analyze.core.llm import LLMSpec

llm = ctx.llm.get_with_spec(LLMSpec('openai', 'gpt-4.1-mini'))
default_generator = ctx.defaults.conversation_query_feature_generator()
custom_generator = attrs.evolve(default_generator,
                                query_generator_model=llm, query_enrich_model=llm)
custom_components = attrs.evolve(ctx.defaults.analyze_components(),
                                 generators=(custom_generator,))
results = await ctx.ops.analyze(..., components=custom_components)
```

Note: `query_generator_model` is the model used to generate features; `query_enrich_model` is the model used to compute those features, during `analyze` and any later `enrich` operation. You can control them separately. 

## Using on-disk LLM cache

Usecase: cache LLM responses, reuse them across multiple runs, and store them on disk.  

LLM calls are slow and expensive. When working with the same datasets and features, these calls often repeat and their results can be cached. 

Caching is configured when the RunContext is created. By default, a small in-memory cache is used. We can instead configure the context to use an on-disk cache:

```python
from agentune.analyze.api import LlmCacheOnDisk

async with await RunContext.create(llm_cache=LlmCacheOnDisk('cache.db', 100_000_000)) as ctx:
    ...
```

The first argument is the path to the cache file; it will be created if it does not exist. The second argument is the maximum size of the cache contents, in bytes. (This size is approximate; the cache file can grow somewhat larger than this.) 

It is not currently recommended to open the same cache file from multiple processes at once.

## Persisting outputs as json and loading them back

Usecase: you want to save e.g. `AnalyzeResults` or a `RecommendationsReport` as a json string or file, and load it back.
Usecase: you want to transfer a value, e.g. the features in `AnalyzeResults`, to a different RunContext instance.

Values of Agentune classes need to be serialized using dedicated methods which an API very similar to the standard library's `json` module:

```python
from agentune.analyze.run.analysis.base import AnalyzeResults

results: AnalyzeResults = await ctx.ops.analyze(...)

json_string = ctx.json.dumps(results)
with open('results.json', 'w') as f:
    ctx.json.dump(results, f)

results2 = ctx.json.loads(results, AnalyzeResults)
assert results2 == results

with open('results.json', 'r') as f:
    results2 = ctx.json.load(f, AnalyzeResults)
    assert results2 == results
```

The only difference from the methods in the standard library's `json` module is that the load and loads methods require the target class (`AnalyzeResults` in this example) and return an instance of that class, not a dict like `json.loads` does.
