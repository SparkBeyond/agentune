import dataclasses
from typing import Any

import attrs

# A poor man's version of the python 3.13 copy.replace.
# Note that attrs only adds __replace__ if running on python 3.13+, so we can't just
# check for __replace__ and call it.

def replace[T](value: T, **kwargs: Any) -> T:
    val2: Any = value # python/mypy#18973
    if attrs.has(type(val2)):
        return attrs.evolve(val2, **kwargs)
    elif dataclasses.is_dataclass(value):
        # This is a different mypy bug from the above
        return dataclasses.replace(value, **kwargs) # type: ignore[type-var]
    else:
        dunder_replace = getattr(val2, '__replace__', None)
        if dunder_replace is None:
            raise NotImplementedError(f'replace is not implemented for {type(value)}')
        else:
            return dunder_replace(**kwargs)
