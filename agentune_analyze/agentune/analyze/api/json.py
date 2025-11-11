import io
from pathlib import Path
from typing import Any, TextIO

from attrs import frozen

from agentune.analyze.api.base import RunContext


@frozen
class BoundJson:
    """Methods for working with JSON data, bound to a RunContext instance.

    Some agentune classes have custom logic for serializing to/from JSON.
    These methods provide the official interface to do so.

    They also support any other values, including dataclasses and attrs classes, that the `cattrs`
    serialization library supports out of the box.
    To customize the serialization logic for additional types, see the `agentune.analyze.core.sercontext` module.
    """
    run_context: RunContext

    def dumps(self, value: Any, **kwargs: Any) -> str:
        """Serialize an object to a JSON string.

        Args:
            value: the object to serialize.
            kwargs: additional keyword arguments to pass to `json.dumps`.
        """
        return self.run_context._ser_context.converter.dumps(value, **kwargs)

    def dump(self, value: Any, target: str | Path | TextIO, **kwargs: Any) -> None:
        """Serialize an object to a JSON string and write it to a file or stream.

        Args:
            value: the object to serialize.
            target: the path or stream to write to.
            kwargs: additional keyword arguments to pass to `json.dumps`.
        """
        match target:
            case str() | Path():
                with Path(target).open('w') as f:
                    f.write(self.dumps(value, **kwargs))
            case _ if isinstance(target, io.TextIOBase):
                target.write(self.dumps(value, **kwargs))
            case _:
                raise TypeError(f'Unsupported target type: {type(target)}')

    def loads[T](self, value: str, cls: type[T], **kwargs: Any) -> T:
        """Deserialize a JSON string to an object of the given class.

        Args:
            value: the JSON string to deserialize.
            cls: the class to deserialize as. Class hierarchies like `Feature` deserialize to the original subclass
                 if you specify the base class (eg Feature) as cls.
            kwargs: additional keyword arguments to pass to `json.loads`.
        """
        return self.run_context._ser_context.converter.loads(value, cls, **kwargs)

    def load[T](self, source: str | Path | TextIO, cls: type[T], **kwargs: Any) -> T:
        """Read a JSON string from a file or stream and deserialize it to an object of the given class.

        Args:
            source: the path or stream to read.
            cls: the class to deserialize as. Class hierarchies like `Feature` deserialize to the original subclass
                 if you specify the base class (eg Feature) as cls.
            kwargs: additional keyword arguments to pass to `json.loads`.
        """
        match source:
            case str() | Path():
                with Path(source).open() as f:
                    return self.loads(f.read(), cls, **kwargs)
            case _ if isinstance(source, io.TextIOBase):
                return self.loads(source.read(), cls, **kwargs)
            case _:
                raise TypeError(f'Unsupported source type: {type(source)}')
