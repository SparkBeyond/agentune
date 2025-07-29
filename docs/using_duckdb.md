# Rules for using duckdb in this codebase

## Naming duckdb databases

As described in duckdb.md, we can control the catalog name of the all attached databases *except* the first one, the one connected to by the `duckdb.connect` call. This leaves two alternatives:

1. Always use the defaults. Manage files so that their basenames are always unique (in any scope where you might want to use several with the same connection). Don't ever use more than one in-memory database, because they're all named 'memory' by default.

2. Always connect (with the original call to `duckdb.connect`) to an in-memory database. Never use it. Attach all the real databases you're going to use (including in-memory ones) with explicitly specified names. Now you can rely on the names.

    The problem with this approach is that the default database setting is connection-scoped. In order to set another database as the default one, you have to execute "USE foo" - every time you call .cursor() (getting the name 'foo' by querying the previous connection), which would add overhead and create a lot of boilerplate and room for mistakes. (We could work around this by wrapping/replacing the original Connection instance with a wrapper whose .cursor() method did this automatically, but this would create secondary problems and introduce complexity.)

So at least for now, we're going with the first approach, and using default database names everywhere. The user (who ultimately specifies on-disk database names, as long as we're not writing service / storage management code) is responsible for not telling us to work with two databases with the same name at once. (We will notice and fail if asked to do so.)

Code MUST NOT do anything that would break if the same database had a different catalog name in a future run. That means you can use the current catalog name in a dynamically constructed query, but you can't e.g. create a custom function in the database that statically refers to the catalog name you saw the first time.

## Managing duckdb connection instances

1. Every connection instance (acquired by calling `.cursor()` either on another connection or on DuckdbManager) MUST be scoped using `with`. That is, all new connections opened in a code scope (=inside a call to a function or an async function) MUST be closed when that call returns. 
    
    This ensures that, after a run completes and the DuckdbManager instance itself is close()d, the database is really closed and all resources are freed (the duckdb threadpool, any in-memory data).

2. Code that passes a connection instance to another thread MUST call .cursor() and pass that instead; a connection instance isn't threadsafe (and is blocking anyway).

3. Code that receives a connection instance as a parameter, and passes it on to some other function, SHOULD just pass it as is; there is no reason to create a new cursor per function call.

4. Code that uses a connection in any way (executing a statement, etc) MUST create a local cursor and use that instead. (A local cursor can be used for successive statements; you SHOULD NOT create a new cursor for each statement in simple sequential code without a reason.) 

5. Relation instances MUST be consumed quickly, deterministically, and exactly once, after being created. They MUST NOT be stored for later use (create them later instead) or passed to code that is complex enough that it might consume them twice accidentally.

    There are valid three patterns of using Relations:

    1. Create and consume it yourself.
    2. Create and return it it on request. This happens in methods like `to_duckdb` (of Dataset, DatasetSource, etc.). Such a function MUST take a Connection parameter, return a Relation backed by that same connection instance (not a cursor), not consume the Relation itself before returning it, and not use that connection in other ways that would manipulate its result set after the Relation is created. (They can create a separate cursor and use it as they see fit.)
    3. Call a function that returns a Relation (as above), passing in a Connection instance. Such code should behave as if it created the Relation itself (consume it promptly, do it all with a cursor, etc).

The scope of "code" (that uses a connection, etc) depends on the good sense of the developer. You don't need to create an extra cursor when you, e.g., refactor one public function into a public function calling several private ones in sequence. What matters is code locality and complexity. Required effects (like closing a connection or consuming a result set) shouldn't rely on distant code (distant code is that which might be changed without considering the local code where you are), or on complex code (where it's not obvious what codepath will be taken, or the code itself is a parameter, like an abstract class or callback).

## TODO (missing docs)

Need to add information on:

- Handling (fully qualified) names in queries (once #43 is resolved)
- Quoting names in queries
- Managing schemas, i.e. the correct ways to move data to/from duckdb (this can be just a list of pointers to the relevant code, which should be documented). Maybe make that a separate 'overview' section that will come first.
- Nonce names (once #42 is resolved)

