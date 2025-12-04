from collections.abc import Sequence
from typing import Any

from frozendict import frozendict
from llama_index.core.base.llms.types import (
    ChatMessage,
)


class CanonicalizedChatMessage:
    """Wrap ChatMessage, which isn't hashable (because it's not frozen), and provide a hash implementation.

    Also make sure all the dicts in the message are sorted (i.e. that the dict's order of iteration is in sorted order of the keys).
    """

    def __init__(self, message: ChatMessage) -> None:
        self.message = self.canonicalize(message)
        self._hash = hash(str(self.message))

    @staticmethod
    def canonicalize(message: ChatMessage) -> ChatMessage:
        return ChatMessage(
            role=message.role,
            additional_kwargs=CanonicalizedChatMessage.sort_dict(message.additional_kwargs),
            blocks=message.blocks
        )

    @staticmethod
    def sort_dict(map: dict[str, Any]) -> dict[str, Any]:
        return { k: map[k] for k in sorted(map) }

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, value: object) -> bool:
        return isinstance(value, CanonicalizedChatMessage) and self.message == value.message

    def __str__(self) -> str:
        return str(self.message)

def _messages_tuple_converter(messages: Sequence[ChatMessage] | Sequence[CanonicalizedChatMessage]) -> tuple[CanonicalizedChatMessage, ...]:
    return tuple(CanonicalizedChatMessage(m) if isinstance(m, ChatMessage) else m for m in messages)


def _try_make_hashable(value: Any) -> Any:
    match value:
        case dict(): return _kwargs_freeze_converter(value)
        case list() | tuple(): return tuple(_try_make_hashable(v) for v in value)
        case _: return value

def _kwargs_freeze_converter(kwargs: Any) -> frozendict[str, Any]:
    """The **kwargs of an LLM method can contain unhashable values like lists and dicts;
    this approach makes it safe to hash and to compare without losing information but doesn't guarantee that
    the result is hashable.
    """
    return frozendict({ key: _try_make_hashable(value) for key, value in kwargs.items() })
