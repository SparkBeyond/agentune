# Rules for using duckdb in this codebase

## Duckdb object names and quoting in queries

'Catalog objects' include tables, views, indexes, etc. 

In the codebase, objects are identified by instances of DuckdbName. A name is always fully qualified, i.e. the database and schema names are known,
as in "database.schema.foo". 

To create a qualified name, you can call `DuckdbName.qualify(object_name, conn)` to use the connection's default database and schema.
High-level methods that accept a name and a connection SHOULD accept a (DuckdbName | str) and, if passed a string,
qualify it using the connection.

When writing queries (for conn.sql() or conn.execute()), always use DuckdbName and not a bare string. Stringify the object directly;
do not add quotes - it will add quotes automatically.

When specifying *column* names in a query, if the names are dynamic (runtime parametrs), you MUST add quotes around them.

Example:

```python
name: DuckdbName = ...
column: str = 'col1'
conn.sql(f'SELECT "{column}" from {name}')
```

## Naming duckdb databases

As described in duckdb.md, we can control the catalog name of the all attached databases *except* the first one, the one connected to by the `duckdb.connect` call. This leaves two alternatives:

1. Always use the defaults. Manage files so that their basenames are always unique (in any scope where you might want to use several with the same connection). Don't ever use more than one in-memory database, because they're all named 'memory' by default.

2. Always connect (with the original call to `duckdb.connect`) to an in-memory database. Never use it. Attach all the real databases you're going to use (including in-memory ones) with explicitly specified names. Now you can rely on the names.

    The problem with this approach is that the default database setting is connection-scoped. In order to set another database as the default one, you have to execute "USE foo" - every time you call .cursor() (getting the name 'foo' by querying the previous connection), which would add overhead and create a lot of boilerplate and room for mistakes. (We could work around this by wrapping/replacing the original Connection instance with a wrapper whose .cursor() method did this automatically, but this would create secondary problems and introduce complexity.)

So at least for now, we're going with the first approach, and using default database names everywhere. The user (who ultimately specifies on-disk database names, as long as we're not writing service / storage management code) is responsible for not telling us to work with two databases with the same name at once. (We will notice and fail if asked to do so.)

Code MUST NOT do anything that would break if the same database had a different catalog name in a future run. That means you can use the current catalog name in a dynamically constructed query, but you can't e.g. create a custom function in the database that statically refers to the catalog name you saw the first time.

## Managing duckdb connection instances

1. Every connection instance (acquired by calling `.cursor()` either on another connection or on DuckdbManager) MUST be scoped using `with`. In particular,all new connections opened in a code scope (=inside a call to a function or an async function) MUST be closed when that call returns. 
    
    This ensures that, after a run completes and the DuckdbManager instance itself is close()d, the database is really closed and all resources are freed (the duckdb threadpool, any in-memory data).

2. Code that passes a connection instance to another thread MUST call .cursor() and pass that instead; a connection instance isn't threadsafe (and is blocking anyway). The code is also responsible for closing the passed cursor when the operation on the other thread completes.

3. Code that receives a connection instance as a parameter and passes it on to other functions but doesn't use it itself SHOULD just pass it as is, without creating new cursors.

4. Code that uses a connection in any way (executing a statement, etc) can normally use the connection passed to it directly. However, it MUST create and use a local cursor instead, IF:

   1. It passes the connection to any other code in the middle of using it itself. (After other code uses a connection, you can't rely on the last result set still being open and unchanged.)
   2. There's a chance it will close the connection (you shouldn't close the original connection passed to you).
   3. It does anything affecting future use of the connection, like setting the default database (executing USE), creating temporary catalog objects, or registering objects for replacement scans.

5. If in doubt, you can always create a local cursor. Creating and closing a cursor is very fast; at least two or three orders of magnitude faster than the simplest query you can run using that cursor.

6. Relation instances MUST be consumed quickly, deterministically, and exactly once, after being created. They MUST NOT be stored for later use (create them later instead). Avoid writing code that accepts a Relation and is complex enough to try to consume it twice accidentally.

    There are valid three patterns of using Relations:

    1. Create and consume it yourself.
    2. Create and return it on request. This happens in methods like `to_duckdb` (of Dataset, DatasetSource, etc.). Such a function MUST take a Connection parameter, return a Relation backed by that same connection instance (not a cursor), not consume the Relation itself before returning it, and not use that connection in other ways that would manipulate its result set after the Relation is created. (They can create a separate cursor and use it as they see fit.)
    3. Call a function that returns a Relation (as above), passing in a Connection instance. Such code should behave as if it created the Relation itself (consume it promptly, do it all with a cursor, etc).

7. When creating in-memory duckdb databases in tests, remember to use `duckdb.connect(':memory:')` to get a new database. Never use `duckdb.connect()` without parameters; this returns a connection to the 'default' in-memory database, which lives as long as the process does. 


The scope of "code" (that uses a connection, etc) depends on the good sense of the developer; it is not always a single function. You don't need to create an extra cursor when you, e.g., refactor one public function into a public function calling several private ones in sequence. 

What matters is code locality and complexity. Required effects (like closing a connection or consuming a result set) shouldn't rely on distant code (distant code is that which might be changed without considering the local code where you are), or on complex code (where it's not obvious what codepath will be taken, or the code itself is a parameter, like an abstract class or callback).

## The temporary database

It is often useful to create temporary catalog objects (tables, views, functions, etc) which are guaranteed to be dropped 
at the end of a scope or the end of a program run. 
Unfortunately, duckdb's native temporary tables are scoped to a Connection instance, which makes them almost useless in our architecture.

We create a dedicated schema in the main database DuckdbManager connects to, and drop it when the manager closes.
We also drop it if it already exists on startup. The schema's name is given by `DuckdbManager.temp_schema_name`;
random object names in that schema should be generated using `DuckdbManager.temp_random_name`.

You MUST drop the temporary objects in a `finally` once you're done with them. DuckdbManager provides a backstop
but dropping them early frees memory and disk space.

In the future, we might prefer to create some or all temporary objects in an in-memory database (with spillover to the duckdb temp 
directory if it runs out of memory). This design is intended to make such a change transparent to any code that uses
`DuckdbManager.temp_random_name`; code that uses `DuckdbManager.temp_schema_name` directly will need to be updated 
to also use a new database name. 

## Other notes

### `sql` vs `execute`

Call `connection.sql()` only when you want to create a Relation instance. Call `connection.execute()` when executing non-query statements whose results you won't consume, and when using prepared parameters.

It's possible to use `connection.sql()` with non-query statements; they are executed immediately without waiting for a method call like .fetchall(). And it's also possible to use `connection.execute()` to run a non-parameterized query (i.e. not a prepared statement). I find this confusing and prefer not to mix the two methods.

### Replacement scans

When using a replacement scan, you MUST explicitly call `connection.register`. Do not use the replacement scan automatic feature of looking up python variables to resolve unfamiliar names.

Make sure the name you use to register the object doesn't shadow any other name you use in queries on that connection. You can use `DuckdbManager.random_name` to generate a nonce name.

## TODO (missing docs)

Need to add information on:

- Handling (fully qualified) names in queries (once #43 is resolved)
- Quoting names in queries
- Managing schemas, i.e. the correct ways to move data to/from duckdb (this can be just a list of pointers to the relevant code, which should be documented). Maybe make that a separate 'overview' section that will come first.
- Nonce names (once #42 is resolved)

