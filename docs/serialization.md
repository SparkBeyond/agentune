# Serialization

We use the [cattrs](https://catt.rs/) library to serialize and validate attrs dataclasses.

This library isn't perfect for our needs, and we may switch to something else in the future. 
(Despite the name, cattrs is not part of attrs.) You don't have to understand cattrs in depth for typical use.

Note that cattrs calls de/serialization 'de/structuring'; we follow this terminology imperfectly in the code.

## Serializing data

### Getting a Converter instance

Serialization is performed by an instance of `cattrs.Converter`. In agentune you can get this instance from the
`SerializationContext`, which is part of the `RunContext`. 

The `SerializationContext` is a stateful class which is necessary to serialize some types (in particular, LLM instances;
see `LLMWithSpec`). 

There is a global `Converter` instance at `sercontext.register_context_converter_hook`, but you should not use it except to register
custom serialization hooks. It is not currently guaranteed to contain all serialization hooks (i.e. to know how to serialize)
all classes, even those that don't require a `SerializationContext` instance.

Serialization outside of a `RunContext` isn't currently well-supported; you can create a `SerializationContext` but 
you shouldn't need to. Please ask for help if this is necessary.

### Serializing data

Data can be converted to and from native Python values acceptable by the json module (dicts and so on) by calling
`Converter.structure` and `Converter.unstructure`. It can also be converted directly to and from JSON strings
by calling `Converter.dumps` and `Converter.loads`; this is equivalent to calling `json.dumps(Converter.unstructure(...))`
etc.

Details: cattrs includes preconfigured JsonConverters that use different json libraries (and not the stdlib json module),
such as orjson. Besides being faster, using these can result in serializing some values differently (eg datetimes, string enums).
Right now, the use of the stdlib json module is hardcoded (in that `sercontext.register_context_converter_hook` is initialized that way);
we could relax this or make it configurable in the future, but would need to be careful. See the 
[cattrs docs](https://catt.rs/en/stable/preconf.html).

## Designing classes for serialization

Classes should match the desired json structure as closely as possible. If the json must look different from the class,
write another class that maps cleanly to the json format and then convert between the classes using python code
(i.e. with attrs.field(converter=...)). This is preferrable to doing the conversion in a custom serialization hook.

We want to avoid custom serialization logic for several reasons. First, it's harder to write and to annotate with types,
because the cattrs APIs aren't as simple as they could be. Second, serialization hooks in cattrs don't compose cleanly;
the order of registration matters and isn't obvious, and there is more complexity than this due to the cattrs implementation.
Third, we don't want to tightly couple to cattrs, and we might replace cattrs in the future.

## Registering custom conversion logic

Custom serialization hooks are unavoidable for new 'primitives' (as opposed to composite types like dataclasses and lists).
('Primitives' are in quotes because they may be classes, not Python primitives.) For example, datetime is represented in 
JSON as a string; if cattrs had not included built-in support, we would have to define a custom serialization hook for
converting datetime to and from str.

Register custom hooks with `sercontext.converter_for_hook_registration`. Prefer simple (single-dispatch) hooks to predicate hooks
to hook factories.

If the custom type is defined in our code, register the hook after the type definition; if it's a library type,
register it in sercontext.py after the definition of `converter_for_hook_registration`; if it might depend on custom
serialization logic registered for other types, such that the order of registration matters, consider calling 
`sercontext.register_context_converter_hook` to have a better chance of applying your hook after others.

## Classes with LLM parameters

We have a dedicated system for serializing LLM instances. See the classes `LLMSpec` and `LLMWithSpec`.

A serializalbe class that needs an LLM instance at runtime should take a parameter of type `LLMWithSpec`. 
When such a class is serialized, the LLMSpec will appear in the JSON data, and the LLMWithSpec (including the LLM instance)
will be repopulated when the class is deserialized.

Note that the `LLMSpec` only specifies the model and provider. If you use non-default model parameters that affect 
its output, you need to deterministically pass the same parameter values on every call (depending on your input),
or else to include those parameters as additional (serializable) class fields. In other words, your class instance
must behave the same (wrt the LLM) after serializing and deserializing it.

## Abstract class hierarchies

Class hierarchies are serialized with an extra JSON field '_type' whose value is the name of the class.
To enable this behavior, the abstract base class must extend the marker class `UseTypeTags`.

If two classes with the same name extend the same base class, serialization will fail.

You can override the value of the _type field by overriding `_type_tag(cls: type) -> str`;
note that this is a class method, not an instance method.
