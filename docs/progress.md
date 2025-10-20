# Progress reporting

An instance of class `ProgressStage` (a 'stage') is a progress indicator which has:
- A name
- A start time, and eventually an end time
- Optionally, a count, which increments over time. Optionally, a total (expected count), which can also be updated.
  
The count is absent in stages with no internal progress (e.g. some kinds of feature selection).
It might also be absent in stages that only serve to group other sub-stages (which can have separate counters), e.g. “feature generation” with a sub-stage for each individual generator. But, where possible, it’s better to publish a counter corresponding to the completion of the sub-stages (e.g. 2/3 of feature generators completed).

Stages are organized in a tree. There is a single root stage (at any one point in time, while running).
The 'current' stage is stored in a context var in the context.base module. This allows code to publish a new stage without knowing where it was called from; the resulting tree of stages corresponds to the call stack.

Because stages are stored in a context var, each thread and callstack can have a separate root. However, that is not how they are meant to be used. 

The intended use is:
1. Every top-level operation in the library is async (and there is only one async thread). All sync operations come from async code calling to_thread, preserving the contextvars.
2. Progress reporters must also be async (at least the part that calls progress.current_stage()); that gives them access to the root stage that was defined by the top-level (async) library operation.

However:
3. If a user calls a synchronous library function directly (e.g. ingesting data into duckdb), that operation will publish progress information, but the user will not be able to consume it in a progress reporter. 

In fact, that progress reporter has nowhere to run concurrently with the synchronous operation - especially if all the progress reporters we implement are async. 

Usecase (3) can still be supported with some extra work:

- User's main (synchronous) thread creates a progress stage, outside of the actual library component to be called
- User instantiates the progress reporter, dispatching it to another thread (whether sync or async), in a way that preserves context vars
- User then calls the synchronous library component

This is not implemented yet, but we may do it in a future wrapper function.  

## Publishing progress information

The simplest way to publish progress information is to write,

```python
total=50
with progress.stage_scope('name', count=0, total=total) as stage:
    for i in range(50):
        do_something(i)
        stage.set_count(i + 1)
```

Note that stage instances are mutable (and thread-safe).

If you don't want to pass around the stage instance (e.g. to avoid adding it to a public API), you can call progress.current_stage() to get it back.

A stage is completed when the context manager exits, or when you call stage.complete(). This sets total=count (if count is not None), sets `stage.completed` to the current datetime and disallows all future updates, including adding child stages. 

Note that child stages can never be removed; stages are only lost when the root stage is removed (e.g. by exiting its context manager).

## Consuming progress information

You can get the current progress state by calling `progress.current_stage().root.deepcopy()`. Deepcopy is needed because the stages are mutable, and you need to get a consistent snapshot.

You can then implement a component that calls this periodically and outputs it to a log or updates an interactive display.

When the root stage is removed (e.g. the context manager that created it ends), all progress information is lost. (The root stage is whichever stage was created first.) If you're going to call a routine that publishes progress information, and the information is going to be lost when the routine returns, you have three choices:

1. Get periodic progress snapshots while the routine runs using a concurrent async loop. You are not guaranteed to get the very first or very last progress state published.
2. Set your own stage wrapping the call to the routine. This makes the last published state persist after the routine returns. However, your progress reporter will need to ignore this outer stage in its reporting. 
3. When using some built-in routines such as feature search, the last published state is also returned (in FeatureSearchResults).

## Working with progress diffs

Because of the allowed updates to a stage, it's possible to compute a difference between any two snapshots of the same progress tree. This computation does not depend on when the snapshots were taken or how many there were; if you 'sum' diffs you will end up at the same final state. This makes it possible to write progress reporters that operate in terms of diffs (i.e. updates) and not in terms of tree snapshots.

Such a progress diff would be a list of events, the possible events being:
- Add a stage to the tree at a location
- Update a stage's count and/or total
- Mark a stage as completed (and set its completion time)

## Other notes

LLM calls cannot be trivially published as a stage with a count. This subject is left for a future PR.

