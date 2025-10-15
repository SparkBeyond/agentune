# Overview of the Agentune architecture

## Library-nature

All Agentune code can be used as a python library. It has no dependencies other than Python packages, and makes no additional assumptions
about the environment it runs in unless told to by user code. It doesn't require any external services or processes, and it doesn't 
run any sub-processes by default.

This is linked to two product requirements:

1. Users can use parts of Agentune as a library, including on local development machines, and without installing anything other than a python package.
2. All parts of Agentune can run anywhere a python library can; this lets us e.g. run on Spark and integrate with various Pythonic frameworks.

### Design implications

1. We are never opinionated. The user can tell us to do anything they want. We don't design our APIs or restrict our code to force or to forbid anything.
2. We store data in files (local and remote) and not any kind of database that needs a standalone server.
3. We scale horizontally, not vertically. Approximately speaking, to work with a huge dataset you need to figure out how to shard it.
4. We never read or write files the user didn't explicitly ask us to, and prefer working with single files to assuming control of entire directories.

## Core libraries

Agentune is written in Python 3.12

### duckdb and polars

We use duckdb for:

- Interacting with external data formats and remote data storages - the data transports underlying ingest and export.
- Storing local copies of datasets to be read again later.
- Indexing local copies of user datasets to speed up queries.
- Copying datasets between nodes (in a future distributed scenario)

We use polars to represent dataframes that fit in memory and perform computations on them. When we want to process data, we usually write python code that operates on a sequence of Polars dataframes, although we can also write an SQL query.

Note that we do NOT use the experimental polars streaming mode, and we do not use polars to read & write any external data formats. Duckdb is better at those things.

### attrs and cattrs

We use attrs to define dataclasses - that is, classes primarily representing data. Dataclasses are immutable (i.e. use `@frozen` not `@define`) by default.

Use of attrs is mandatory for dataclasses. It is optional (but widespread) for classes where it merely provides a more convenient syntax and
saves boilerplate. Make sure to pass `eq=False` and/or `hash=False` where appropriate.

We use cattrs to serialize (aka un/structure) attrs dataclass instances; see [serialization.md]() for more details.

## Types and schemas

We have our own classes `Dtype`, `Field` and `Schema`. They map to both duckdb and polars types, and it's possible to convert between all three.

We don't use e.g. Polars' `DataType`, `Field` and `Schema`, because:

- Defining our own `Dtype` lets us restrict the types supported by Agentune, and to determine the physical representation of some logical types that can be represented in multiple ways.
- We will want to store more schema metadata in the future (e.g. semantic subtypes), which cannot be stored natively by duckdb or Polars.
- The builtin methods converting between duckdb result sets and polars dataframes lose some type information (eg for enums), so we provide our own conversion code.

## Datasets

A `Dataset` is essentially a dataframe, with extra schema information.

A `DatasetStream` is a readable-once iterator or asynchronous iterator of dataset chunks, with a known schema.

A `DatasetStreamSource` can be opened many times to provide separately consumable `DatasetStream`s.

A `DatasetSink` can write a `DatasetStream` to some destination.

The classes `DatabaseTable` and `DatabaseIndex` help declare, create and analyze duckdb table schemas.

## Code modularity

We strive to make code as modular as possible. This generally means separating interfaces from implementations,
and having all other code work with the interface and not assume on a particular implementation.

It is expected that the user (or another library) can introduce a new implementation of any interface. 
To this end, interfaces must be fully documented, so that it is possible to tell whether a new implementation is valid or not.

Some interfaces can have sync and async implementation variants (see [threading.md]()). 

## Code organization

### Private code

Code is public by default: that is, meant for library users, well documented and typed, and with breaking changes noted in release notes. 

Code can be marked private with a note in a module or package docstring. Marking a package private includes all modules and sub-packages in it.

Symbols starting with _ are always private, as per Python convention.

Classes should *not* document different rules for determining which attributes are public and can be depended on.

(TODO: marking some code as private has not yet been done consistently throughout the codebase.)

### Interfaces and implementations

Interfaces must extend `abc.ABC`, and use the `@abstractmethod`, `@final`, etc. family of decorators.
Implementations must specify `@override` where possible.

Interfaces should not use attrs `@define`; instead they have to define properties with `@property @abstractmethod def foo(self) -> T: ...`.
This allows implementations not to use attrs.

We are considering changing this rule in the future, since using `@define` would make some interfaces much more readable, 
and it is possible to implement such an interface without using attrs (or to work around any shortcomings of attrs)
by manually writing out the appropriate `__init__` and other dunder methods.

By convention, we put interfaces in a module called `base.py` in each package, and their implementations in other modules in the same package.
This is not an absolute rule, since some packages have too many (unrelated) interfaces to place them all in a single base module.

For interfaces which might have an async implementation, see also threading.md.