# Writing async components

Each code component (loosely defined) is either synchronous or asynchronous. 

Components which have an abstract interface and one or more implementation always declare the interface ABC to be async,
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
3. Calls which technically block but are guaranteed to be very short can sometimes be allowed in async code, if it's a strong guarantee
   and dispatching it to sync code has a clear downside, but this is a rare exception and you should never decide to do it
   without exhausting every alternative.

## Other rules for async components

All async components run on the same async thread, together with other async libraries, and need to be well-behaved.

They must not compute things for a long time without either yielding (`await sleep(0)`) or sending the computation
to a sync thread.

# High-level runners

Runners are the high-level functions that compose multiple components together. A typical example is running several feature generators
and selecting the best features among them.

A runner needs to handle each component (implementation) being sync or async, and discovering which only at runtime.

Runners are always async (or wrappers calling asyncio.run on the real async runner). During a running operation,
there is a single async thread and a threadpool of sync threads to which sync code can be dispatched. 
This follows normal asyncio convention.

# Thread safety

## Preface

Python very few documented rules for thread safety. Primitives (float, int, string, bool, None, tuples, ...) are threadsafe;
reading and assigning local variables is threadsafe; anything else is not guaranteed.

Some things are considered safe "in practice", as CPython implementation details - like readonly access to dicts 
(which includes access to class members!) - but this is deliberately not a documented guarantee.

Also, if you don't fully control the code behind the class you're sharing, it might involve a write operation 
under the hood (setting a private value, perhaps to cache something) as part of reading a property.

For user-defined classes, we should only share our best bet is to only share readonly classes whose definition we control, 
and which are implemented with slots (the default for attrs dataclasses).

If you need to share anything other than that, it must be explicilty handled in one way or another.

## Known cases requiring handling

1. Dataframes and series are explicitly thread-unsafe. However, the underlying Arrow data is immutable and threadsafe.
   Call .clone() on a df or series and share the result; it is a very cheap zero-copy operation creating a new class
   wrapping the same data.
2. Duckdb connection instances (DuckDBPyConnection) *are* threadsafe, but using them is blocking, so multiple
   threads can't use the same connection. Call .cursor() to cheaply get a new connection (with the same attached databases)
   and send that to the other thread. Remember to close the new connection when done (as with all calls to .cursor()).

