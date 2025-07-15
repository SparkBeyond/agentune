# Register hooks after loading modules in the correct order, to avoid a circular import.
from agentune.analyze.util import cattrutil

from . import llm, sercontext  # noqa: F401 unused imports

cattrutil.configure_lazy_include_subclasses # noqa: B018 useless expression
