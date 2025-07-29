# Notes and explanations about duckdb

This is a collection of my notes about duckdb's behavior, documenting some things that I found 
difficult to understand from the existing duckdb documentation. Some of it comes from reading various
duckdb tickets and discussions, some from experimentation.

This assumes thorough knowledge of what duckdb already documents; both the general and the python-specific
sections. This is *not* an introduction to duckdb, or something to read after you've read just a few pages
and examples of the duckdb docs. It is meant to supplement and to underscore things that are especially
relevant to this codebase.

You need to understand the duckdb docs (and then this document) to work with most parts of this codebase.
Yes, there's a lot of it, and you don't know which parts you'll need, so it's tempting to skip it all until
you do need it. Sorry, but there are no shortcuts.

Note that the duckdb python API implements the Python standard DB-API 2.0 (documented [here](https://peps.python.org/pep-0249/)), but the names here can be a bit confusing. A `DuckDBPyConnection` is both a Connection and a Cursor, in terms of the DB-API; the `connection.cursor()` method returns the same type as the original `connection`.  In the rest of this document, I use the term `connection` meaning a `DuckDBPyConnection`, not the narrow interface called a Connection in the DB-API.

I've been wrong before, more than once, about duckdb's behavior. You should trust-but-verify this document,
and please do correct it if you can. You can submit additions, although we don't want to repeat here anything
that is clearly documented in the duckdb python API docs or SQL docs.

## duckdb's management of connections

This is well documented in the duckdb native (C++) API, but that information is not repeated in the python API docs.
The Python API also manages resources and abstractions somewhat differently from the native API, and has a slightly
different feature set. A deep understanding of duckdb includes modeling what the python API is doing in terms of
the underlying native library; here are a few (relevant) things that it's doing.

1. Every call to `duckdb.connect(path)` for a new database path creates a new 'database instance' (in the native API's terms). Every database instance has a (native) threadpool (whose size can be changed later), and some 
DB-instance-level settings and state. These are freed only when all connections to the database instance have been closed.
    
    Calling `duckdb.connect(path)` for a path that already has a live connection in the process returns a new connection to the same database instance; this works until all connections to the database instance are explicitly closed or are garbage collected.

2. Calling `duckdb.connect(f':memory:{name}')` creates a connection to a *named in-memory database*. This is a python API feature that doesn't exist in the native API. The database created isn't "aware" of its name; that is, its catalog is still named 'memory' (the same as regular, anonymous in-memory databases), and no SQL query will return the name you used.

    Subsequent calls (with the same {name}) return connections to the same database. Once all connections have been closed, the database is discarded and the next connection attempt will create a new, empty in-memory database under that name.

    This codebase doesn't use named in-memory databases (unless the user passes us one manually), and doesn't rely on the ability to reconnect to an in-memory database, because they lack some features that would make them useful. Don't use them by accident.

    Note that executing "ATTACH DATABASE ':memory:name'" results in creating or opening an on-disk database in a file named ':memory:name', not an in-memory database, because named in-memory databases are implemented in the duckdb python API, not in the library itself.

    'Normal' (unnamed) in-memory databases used by our code should be created using the string ':memory:' (without a name after the second colon), to keep compatibility with the SQL ATTACH DATABASE api. Such databases are always new ones; you can't reconnect to one of them or attach one to a new connection.

2. Once you have a connection, you can call its .cursor() method to get another connection to the same DB instance. (`DuckDBManager.cursor()` returns a new cursor from its internally held connection.) This is cheap and you should do this when in doubt. However, always enclose a cursor in a `with` block to close it when you're done (see detailed rules below).

    All duckdb operations are blocking, so at minimum we need a cursor per thread. Also, some things happen at connection 
    scope, and it's useful to create connections to scope various effects, like USE statements and transactions.

    Therefore, all code that sends a connection (`class DuckDBPyConnection`) instance across threads MUST call its `.cursor()` and send that value instead.

    Because a relation (`class DuckDBPyRelation`) is linked to a connection instance, relations MUST NOT be sent to other threads. 

3. More databases can be attached to an existing connection; this does not create an additional threadpool.
    
    Attaching/detaching databases affects existing connections created via cursor() calls from each other; only connections created by calling duckdb.connect() are unaffected. (Such connections can still share the database instance.)

    Existing in-memory databases can't be attached to another connection, but multiple in-memory databases can be created and attached to the same connection under different catalog names.

4. Databases don't store their own catalog name inside the database; the catalog name is determined when connecting to them (but see below).

    The original database connected to by `duckdb.connect()` always uses the duckdb default as its catalog name. This is "memory" for in-memory databases and the file basename for file-backed ones. The catalog name for a connected database can't be changed later (but it can be connected to a different connection under a different name, if the connections are readonly).

    When attaching additional databases (using the SQL ATTACH DATABASE statement), it's possible to specify a custom catalog name, overriding that default. (This is reflected in our DuckdbManager API.) 
    
    See below for the consequences on our pattern of using duckdb databases.

## Relations

A relation (`class DuckDBPyRelation`) represents a query that has been parsed and bound; there are no syntax errors, all names have been resolved, and the result schema is known. The query hasn't been run yet, but you can rely on it being valid. (Read the "Relational API" of the Python docs for more details; these are, as usual, only notes on things that were not clear to me after reading it.)

A relation instance can be consumed only once; if you call e.g. `relation.fetchall()` or `relation.to_xxx()` more than once, subsequent calls will silently return empty results with the same schema. This is unfortunate; it means a relation doesn't really represent a (repeatable) query. It's also not obvious how to tell whether a relation has already been (partially or fully) consumed.

A further point of confusion is that `rel.show()` (and also `str(relation)` and `repr(relation)`) will fetch and cache the first 10K rows of the relation's results, but will print or return (as a string) only a small sample of those rows. This can be done multiple times, both before and after (and during) consuming the relation, and does not affect it; the 10K rows are cached transparently on the Relation instance. (Code outside an interactive shell normally has no reason to use this preview, and shouldn't.) 

This can be confusing when looking at when rows are actually fetched or computed (e.g. by calling a function in the query) versus when you get the results from the Relation instance. It also means that a Relation instance, after its results being consumed, holds an in-memory cache of the first 10K rows of its results, even if you never asked for it.

Therefore, code MUST only ever create a Relation that will definitely be consumed quickly, in a close-by and known location, exactly one. It's bug-prone to e.g. store a Relation for later use, or to pass it somewhere that will not immediately consume it.

A Relation instance is backed by the Connection instance that created it. If the Connection is closed, the Relation cannot be consumed. This is another reason not to store Relations, and the result set semantics (below) provide yet another reason not to do it.

## Result sets

'Result set' is not a formal duckdb term; it is the equivalent term used in JDBC and other APIs. I haven't found a consistent duckdb name for this concept. It is the thing that a "cursor" is a pointer (=cursor) into.

There is at most one result set per a connection instance. A new result set is created whenever you execute a new SQL statement; the previous result set, if any, is automatically discarded. (An open result set can also be explicitly discarded without consuming the remaining results.)

Relations created from a connection instance also use (and replace) its result set. If you consume results piecewise (eg with `.fetchmany()` but not with `.fetchall()`), and interleave those calls on two relations built from the same connection, the results you will get are undefined (and, in practice, always wrong). The same thing hapens if you interleave calls on the connection itself (`connection.execute(query_string)` followed by `connection.fetchmany()`) with calls on a relation backed by that connection.

Duckdb always fetches whole result chunks (this a technical term; a chunk is 8k rows by default), even if you call `.fetchone()` or call `.fetchmany()` with a number smaller than 8k. If you consume fewer than 8k rows, the rest are cached and will be returned on the next call. This cache is stored on the relation instance (if you're using one) and not on the connection instance itself, so if you try the interleaving experiment with small numbers of rows, you will get apparently correct results. This does not work with more than 8k rows (total, per query) and code MUST NOT interleave reading relations - there is only ever one active result set on the connection.
