from cattrs.preconf.json import make_converter
from datetime import timedelta

# This converter instance should be used to un/structure all classes from this library.
# If custom hooks are needed in the future, they will be added here.

converter = make_converter()

@converter.register_unstructure_hook
def _unstructure_timedelta(td: timedelta) -> float:
    return td.total_seconds()

@converter.register_structure_hook
def _structure_timedelta(d: float, _) -> timedelta:
    return timedelta(seconds=d)
