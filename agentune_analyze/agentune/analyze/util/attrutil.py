from collections.abc import Mapping

from frozendict import frozendict


def frozendict_converter(input: Mapping) -> frozendict:
    """Use this function as the converter= argument to attrs.field.

    Using converter=frozendict fails because mypy (wrongly) doesn't understand that
    frozendict can take a dict as input.
    """
    return frozendict(input)

