# Rules for using duckdb in this codebase

## Duckdb object names and quoting in queries

'Catalog objects' include tables, views, indexes, etc. 

In the codebase, objects are identified by instances of DuckdbName. A name is always fully qualified, i.e. the database and schema names are known,
as in "database.schema.foo". 

To create a qualified name, you can call `DuckdbName.qualify(object_name, conn)` to use the connection's default database and schema.
High-level methods that accept a name and a connection SHOULD accept a (DuckdbName | str) and, if passed a string,
qualify it using the connection.

When writing queries (for conn.sql() or conn.execute()), always use DuckdbName and not a bare string. Stringify the object directly;
do not add quotes - the implementation of `DuckdbName.__str__` will add quotes automatically.

When specifying *column* names in a query, if the names are dynamic (runtime parameters), you MUST add quotes around them.

Example:

```python
name: DuckdbName = ...
column: str = 'col1'
conn.from_sql(f'SELECT "{column}" from {name}')
```

## Naming duckdb databases

WARNING: working with multiple databases is not yet fully supported and the documentation here may change in the future; it describes the current plans.

As described in [duckdb.md](duckdb.md), we can control the catalog name of all attached databases *except* the first one, the one connected to by the `duckdb.connect` call. 

I tried working around this by connecting to an in-memory database that is never used, and then attaching all the databases we want to use under the desired names.
Unfortunately, the default database (the thing changed by the SQL statement `USE foo`) is connection-scoped. We call .cursor() very freely, so we would need to e.g. wrap the `DuckDBPyConnection` in a class that remembered the default database and restored it by calling `USE` for each new cursor, but this would be expensive and complicate the code.

The user must treat the first database connected to specially (i.e. to choose which to connect to first and which to attach later). The code never changes the default database. It addresses the main database implicitly (i.e. with bare table names) and any secondary databases explicitly (i.e. with fully-qualified names containing a catalog and schema). 

Bare names (strings) are converted to fully-qualified DuckdbName instances (by `DuckdbName.qualify`), which capture the actual name of the main database, and these DuckdbNames are then captured as attributes of other classes, which may undergo serialization. This will break if, on the next run when these values are deserialized, the main database has a different name.

So code must not do anything that would break if the same database had a different catalog name in a future run. That means you can use the current catalog name in a dynamically constructed query, but you can't e.g. create a custom function in the database that statically refers to the catalog name you saw the first time. DuckdbNames that are serialized should not be ones that refer to the main database.

## Managing duckdb connection instances

1. Every connection instance (acquired by calling `.cursor()` either on another connection or on DuckdbManager) must be scoped using `with`. In particular, all new connections opened in some scope (e.g. inside a call to a function or an async function) must be closed when that call returns. 
    
    This ensures that, after a run completes and the DuckdbManager instance itself is close()d, the database is really closed and all resources are freed (the duckdb threadpool and any in-memory data).

2. Code that passes a connection instance to another thread must call .cursor() and pass that instead; a connection instance isn't threadsafe. (Even if it was threadsafe, two threads using the same connection would block each other.) The code that calls .cursor() is responsible for closing the passed cursor when the operation on the other thread completes.

3. Code that receives a connection instance as a parameter and passes it on to other functions but doesn't use it itself should just pass it as is, without creating new cursors.

4. Code that uses a connection in any way (executing a statement, etc) can normally use the connection passed to it directly. However, it must create and use a local cursor instead if:

   1. It passes the connection to any other code in the middle of using it itself. (After other code uses a connection, you can't rely on the last result set still being open and unchanged.)
   2. There's a chance it will close the connection (you shouldn't close the original connection passed to you).
   3. It does anything affecting future use of the connection, like setting the default database (executing USE), creating temporary catalog objects, or registering objects.

5. If in doubt, you can always create (and then close) a local cursor. Creating and closing a cursor is very fast, orders of magnitude faster than the simplest query you can run using that cursor.

6. Relation instances must be consumed exactly once, deterministically, quickly after being created. They must not be stored for later use (create them later instead). Avoid writing code that accepts a Relation and is complex enough to try to consume it twice accidentally. (See [duckdb.md](duckdb.md) for details.)

    There are valid three patterns of using Relations:

    1. Create and consume it yourself.
    2. Create and return it on request. This happens in methods like `to_duckdb` (of Dataset, DatasetSource, etc.). Such a function must take a Connection parameter, return a Relation backed by that same connection instance (not a cursor), not consume the Relation itself before returning it, and not use that connection in other ways that would manipulate its result set after the Relation is created. (It can create a separate cursor and use it for its own purposes.)
    3. Call a function that returns a Relation (as above), passing in a Connection instance. Such code should behave as if it created the Relation itself (consume it promptly, do it all with a cursor, etc).

7. When creating in-memory duckdb databases in tests, use `duckdb.connect(':memory:')` to get a new database. Never use `duckdb.connect()` without parameters; this returns a connection to the 'default' in-memory database, which lives as long as the process does and exposes your test data to other tests. (Most tests should use one of the fixtures `conn: DuckDBPyConnection`, `ddb_manager: DuckdbManager` or `run_context: RunContext` instead of calling `duckdb.connect` directly.) 

The scope of "code" (that uses a connection, etc) depends on the good sense of the developer; it is not always a single function. You don't need to create an extra cursor when you, e.g., refactor one public function into a public function calling several private ones in sequence. 

What matters is code locality and complexity. Required effects (like closing a connection or consuming a result set) shouldn't rely on distant code (distant code is code which might be changed without considering the local code where you are), or on complex code (where it's not obvious what codepath will be taken, or the code itself is a parameter, like an abstract class or callback).

## The temporary database

It is often useful to create temporary catalog objects (tables, views, functions, etc) which are guaranteed to be dropped 
at the end of a scope or the end of a program run. 
Unfortunately, duckdb's native temporary tables are scoped to a Connection instance, which makes them almost useless in our architecture.

We create a dedicated schema in the main database DuckdbManager connects to, and drop it when the manager closes.
We also drop it if it already exists on startup. The schema's name is given by `DuckdbManager.temp_schema_name`;
random object names in that schema should be generated using `DuckdbManager.temp_random_name`.

You must drop the temporary objects in a `finally` once you're done with them. DuckdbManager provides a backstop
but dropping them early frees memory and disk space.

In the future, we might prefer to create some or all temporary objects in an in-memory database (with spillover to the duckdb temp 
directory if it runs out of memory). This design is intended to make such a change transparent to any code that uses
`DuckdbManager.temp_random_name`; code that uses `DuckdbManager.temp_schema_name` directly will need to be updated 
to also use a new database name. 

## Other notes

### `sql` vs `execute`

Call `connection.sql()` only when you want to create a Relation instance. Call `connection.execute()` when executing non-query statements whose results you won't consume.

It's possible to use `connection.sql()` with non-query statements; they are executed immediately without waiting for a method call like .fetchall(). And it's also possible to use `connection.execute()` to run a non-parameterized query (i.e. not a prepared statement). I find this confusing and prefer not to mix the two methods.

### Replacement scans

When using a replacement scan, you must explicitly call `connection.register`. The replacement scan automatic feature of looking up python variables to resolve unfamiliar names is disabled, and you should not enable it on a connection.

Make sure the name you use to register the object doesn't shadow any other name you use in queries on that connection. You can use `DuckdbManager.random_name` to generate a nonce name.

