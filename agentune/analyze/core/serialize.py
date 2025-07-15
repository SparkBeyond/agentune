import cattrs
import cattrs.preconf.json

# This is a separate module to avoid circular imports.

# Custom hooks that don't need a SerializationContext should be registered directly with this converter.
default_converter: cattrs.Converter = cattrs.preconf.json.make_converter()

