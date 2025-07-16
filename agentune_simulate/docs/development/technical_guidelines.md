**Technical guidelines (scope: library only)**

# TOC {#toc}

[TOC](#toc)

[Abstract interfaces & private implementations](#abstract-interfaces-&-private-implementations)

[Defning entities (dataclasses)](#defning-entities-\(dataclasses\))

[Validation & serialization](#validation-&-serialization)

[Collection types](#collection-types)

[Using type hints](#using-type-hints)

[Async code](#async-code)

[When](#when)

[How](#how)

[Multithreading](#multithreading)

[Library parameters and fixtures](#library-parameters-and-fixtures)

[Representing and storing data](#representing-and-storing-data)

[Other infra code](#other-infra-code)

# Abstract interfaces & private implementations {#abstract-interfaces-&-private-implementations}

APIs (the surface of the library that users should use) is public code and should be separated from private implementation code.

Public vs. private separation is done in two ways:

1. Interface from implementation (can be used even when there’s only one implementation); they can also go in different modules, if the library is large enough to need that  
2. Private symbol names start with \_ or \_\_

An interface is a a public, mostly abstract class. Interfaces should use the \`abc\` module, extending \`abc.ABC\` and using the \`@abtractmethod\` etc. annotations.

```py
class Interface(abc.ABC):
  @abstractmethod
  def foo(self) -> int: …
  
  @property
  @abstractmethod
  def prop(self) -> str: ...

@attrs.define
class Implementation(Interface):
  prop: str = 'hello'

  @override
  def foo(self) -> int: return 42 
```

# Defning entities (dataclasses) {#defning-entities-(dataclasses)}

Use the attrs library, not stdlib dataclasses or Pydantic models.

Writing classes like this:

```py
@define
class Foo:
  bar: int
  baz: str = 'hello'
```

Is a very convenient syntax, which also provides implementations of equality, str(), etc; you’re encouraged to use it.  
(Note that you can use \`attrs.evolve\`, it works like scala’s .copy or Python 3.13’s \`copy.replace\`.)

You are **required** to use it when defining entities, that is, concrete dataclasses that are part of the public API.

Entities should be immutable: use @attrs.frozen not @attrs.define.  
If your class has additional mutable state (not declared as attributes) then document it clearly and make sure equality, hashing and evolve() work correctly.

Entities’ attribute types should also be comparable and hashable, to let the containing class do those things.

### Validation & serialization {#validation-&-serialization}

To define validation or implement (custom) json serialization, use the cattrs library. (I’m not sure OS1, as a pure library, needs to deal with that.)

Note on serialization: in hierarchies, where you need to deserialize the right subclass or the right implementation of an interface:

1. Use untagged unions if possible in hierarchies of concrete classes  
2. Use tagged unions when necessary, particularly for abstract interfaces with multiple implementations  
3. When the list of possible implementations is not STATICALLY known (eg there might be more pluggable implementations at runtime): halt, melt, catch fire, ask Daniel

# Collection types {#collection-types}

In python, lists do not equal tuples, i.e. \[1,2,3\] \!= (1,2,3).   
Furthermore, lists, dicts and sets are not hashable and so cannot be placed in sets or used as dict keys.

Therefore, do NOT use the list type in signatures.  
For function/method inputs, use the Iterable abstract type; for outputs, use either Iterable or tuple.  
For the types of attributes in data classes, always use tuple, not Iterable. (Using Iterable lets you construct a class instance holding a list, which does not compare equal to a tuple.)

The equivalent for set is frozenset, in the standard library, and theSet abstract type.

The equivalent for dict is frozendict, a widely used third party library, and the Mapping abstract type.

# Using type hints  {#using-type-hints}

Use the type hints recommended as of Python 3.12. Type hints in Python keep evolving with Python versions, including syntactically. Some old code styles are deprecated and should not be used. 

Among other things:

1. Don’t use deprecated aliases in the typing module that refer to concrete types; e.g. use \`list\` not \`typing.List\`.  
2. Don’t use deprecated aliases in the typing module that refer to abstract types; e.g. use \`collections.abc.Iterable\` not \`typing.Iterable\`.  
3. Use the new syntax for generic classes (\`class Foo\[T\]\`), don’t extend Generic\[T\]  
4. Prefer type bounds (\`class Foo\[F: Bar\]\`) to explicit TypeVar declarations.   
5. Don’t use string type annotations for forward references; always do \`from \_\_future\_\_ import annotations\` if necessary.

# Async code {#async-code}

This assumes you understand how async/await and asyncio work in python\! Read the docs.

### When {#when}

Code talking to external services (including LLMs, subprocesses, other HTTP servers…) is always async. Code *using* that code is also async.

Never block a thread waiting for some async operation if you can help it.  
If you need to call async code from deep inside sync code, that’s a sign of bad design, ask for design help with your codebase instead of blocking.

### How {#how}

In general, use asyncio (i.e. not anyio or other libraries). Use all the stdlib abstractions it comes with as needed: queues, locks, context vars, task groups, cancellation, etc.

Have only one thread running an asyncio loop.   
The loop should be run (calling asyncio.run) by user code, if you’re writing a library, or by the entrypoint of a service. Never call asyncio.run in your own (library) code, except in tests.

This lets you ignore thread safety in purely async code, and use code (including from the asyncio library) that relies on the ‘current’ event loop and context vars.

Never block in async code. Don’t run long computations either, because you might share the event loop with other code (from outside your library).   
Send both blocking and long-running code to the sync threadpool, by calling asyncio.to\_thread.

### Multithreading {#multithreading}

If you send sync or blocking code to other threads, or if you write multithreaded code regardless of async, be careful of thread safety.  
You must not pass thread-unsafe values between threads, in callbacks or their closures.

Python values that are not primitives, or explicitly documented to be threadsafe (like queues), are NOT threadsafe by default.  
Example: dataframes are NOT threadsafe (this is true both in pandas and in polars).  
So are half your classes and most of the third party libraries you use.

If your code has this concern (i.e. it is at all multithreaded) you MUST be aware of thread safety.   
Ask for help if you need it (both design and implementation).

# Library parameters and fixtures {#library-parameters-and-fixtures}

Some things (‘fixtures’) need to be created or modified by the user (and not the library itself), and passed as arguments to the library’s public API.   
You can define defaults in the public API methods, but you must accept overriding values from the user.   
If two fixtures depend on one another, you have to support the user replacing one by not the other \- e.g. with a builder pattern.

An Important and typical example: an OpenAI client instance needs to be provided by the user (because it may be configured to use the user’s API key or to access a custom server).  
An httpx connection pool (used for all http access) also needs to support providing / overriding by the user, but unlike the OpenAPI client, it can have a default value. However, there should be only one httpx (pool) instance (per asyncio thread), and the OpenAI client instance should use that pool (and not create its own).

# Representing and storing data {#representing-and-storing-data}

This is a brief outline only. The aoa repo has much more detail including implementation bits.  
The OS1 library might not need dataframes, or only need simple ones, etc. so I’m not detailing this for now. Let me know what you end up using.

1. Use the polars library for dataframes, and for calculations on numeric data.  
2. The aoa repo has a detailed list of the column types we support (not the same things as “that polars supports”) and how we handle them.   
   If you only need strings, bools, and numbers you should be fine (do note that you’re using pyarrow dtypes)  
   If you need eg categorical types then be slightly careful; read the polars docs and look at how aoa does it.  
   If you end up doing anything nontrivial, we may need to make the infra code from aoa available to this library.  
3. Use duckdb for 1\) data storage 2\) large data streaming and querying 3\) reading & writing remote sources and various file formats.  
   The aoa repo has documentation and code for converting between duckdb and polars dataframes.  
4. You can use polars and pyarrow for reading files instead, to keep the code simpler, and that’s enough if all you need are tests.   
   Do note that duckdb has a better CSV sniffer. We should NOT invest any effort in improving or working around other libraries’ CVS sniffers.

# Other infra code {#other-infra-code}

We might need a generic LLM query cache, with optionally on-disk storage. This should be a shared component and not an internal part of this library, we’re going to need it everywhere.