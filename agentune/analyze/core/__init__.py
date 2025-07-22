"""The core APIs and functionality used by all of agentune.

Module separation (avioding circular imports):

- setup: functions to initialize the library; everything else can depend on it.
- types: class Dtype; definition and translation between duckdb and polars; everything that comes after this can depend on it.
- schema: class Schema; everything that comes after this can depend on it.

- database: classes DuckdbTable, DuckdbIndex, DuckdbManager.
- dataset: classes Dataset, DatasetSource, DatasetSink.
- duckdbio: implementations of DatasetSource and DatasetSink using duckdb, and other classes like SplitDuckbTable.
            Depends on database and dataset.

- llm: classes LLMSpec, LLMContextManager; code to serialize LLM specs and deserialize them into live LLM instances
- sercontext: class SerializationContext; cattrs-based serialization that uses LLMContext to serialize LLM references
              (Future custom hooks will also be registered here.)
              Depends on llm.
"""

# Register hooks after loading modules in the correct order, to avoid a circular import.
from agentune.analyze.util import cattrutil

from . import llm, sercontext  # noqa: F401 unused imports

cattrutil.configure_lazy_include_subclasses # noqa: B018 useless expression
