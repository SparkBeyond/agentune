# Writing async components

Each code component (loosely defined) is either synchronous or asynchronous. 

Components which have an abstract interface and one or more implementation always declare the interface ABC to be async
and add a SyncXxx subclass which adds abstract synchronous methods and overrides the asynchronous ones to call them,
like this:

```python
class MyComponent(ABC):
    @abstractmethod
    async def ado(self, params): ...

class SyncMyComponent(MyComponent):
    @abstractmethod
    def do(self, params): ... # Same signature as `ado` but not async

    @override
    async def ado(self, params):
        return await asyncio.to_thread(self.do, params)
```

When passing parameters to another thread, as in a call to `asyncio.to_thread`, you must copy thread-unsafe values (see below).

## When to be sync or async

1. Code that uses network / llm calls, or other naturally-asynchronous things like sleeping / waiting for things, must be async.
2. Code that blocks - waiting for IO or another syscall - or that performs long computations (possibly using duckdb or polars),
   must be sync. If an async component needs to run such code, it must dispatch it to a sync thread.
3. Calls which technically block but are guaranteed to be very short (e.g. guaranteed-short duckdb queries like checking a table schema) 
   are allowed in async code, but dispatching them to a sync thread is never an error and you should do it if in any doubt.

## Other rules for async components

All async components run on the same async thread, together with other async libraries, and need to be well-behaved.

They must not compute things for a long time without either yielding (`await sleep(0)`) or sending the computation
to a sync thread.

A 5-second computation embeded in async code doesn't technically block, but it prevents e.g. network operations from 
taking place during those 5 seconds, which can cause timeouts and other problems.

# Components composing other components or features

These are components like AnalyzeRunner and EnrichRunner. They take other components as arguments, which may be async or sync.
In particular, any component that computes Features either has to deal with this or (more commonly) uses EnrichRunner to do it.

Such code needs to handle each sub-component (implementation) being sync or async, and discovering which only at runtime.
The code itself must be async, and dispatch sync sub-components to a sync thread as needed.

# Thread safety

## Preface

Python very few documented rules for thread safety. Primitives (float, int, string, bool, None, tuples, ...) are threadsafe;
getting and setting dict members (including local variables) is threadsafe, if the builtin dict is used and not another wrapper
Mapping class; anything else is not guaranteed.

Some other things are considered safe "in practice", based on CPython implementation details, but are not documented guarantees.

If you don't fully control the code behind a class, even benign-looking read access might involve a write operation 
under the hood (e.g. setting a private value to cache something) as part of reading a property, which may not be threadsafe.
(This is why Polars dataframe instances are not threadsafe even though they are immutable, and Pandas dataframes are not threadsafe
even if used in a non-mutating way.) 

For user-defined classes, we should only share our best bet is to only share readonly classes whose definition we control.
It is better to use slots classes (the default for attrs dataclasses) than dict classes, because there are non-atomic
operations on dicts.

If you need to share anything other values between threads, thread safety must be explicilty handled in one way or another.

## Known cases requiring handling

1. Dataframes and series are explicitly thread-unsafe. However, the underlying Arrow data is immutable and threadsafe.
   Call .clone() on a df or series and share the result; it is a very cheap zero-copy operation creating a new class
   wrapping the same data.
2. Duckdb connection instances (DuckDBPyConnection) *are* threadsafe, but using them is blocking, so multiple
   threads can't use the same connection. Call .cursor() to cheaply get a new connection (with the same attached databases)
   and send that to the other thread. Remember to close the new connection when done (as with all calls to .cursor()).

