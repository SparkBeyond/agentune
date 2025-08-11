import logging
from pathlib import Path

import polars as pl
import pytest
from tests.agentune.analyze.flow.feature_search.toys import (
    ToyAsyncEnrichedFeatureSelector,
    ToyAsyncFeatureGenerator,
    ToySyncFeatureGenerator,
    ToySyncFeatureSelector,
)

from agentune.analyze.context.base import TablesWithContextDefinitions
from agentune.analyze.core.database import DuckdbManager, DuckdbName, DuckdbTable
from agentune.analyze.core.dataset import DatasetSource
from agentune.analyze.core.duckdbio import (
    DuckdbTableSink,
)
from agentune.analyze.run.base import RunContext
from agentune.analyze.run.feature_search.base import (
    FeatureSearchInputData,
    RegressionFeatureSearchParams,
)
from agentune.analyze.run.feature_search.simple import SimpleFeatureSearchRunner
from agentune.analyze.run.ingest import sampling

_logger = logging.getLogger(__name__)


@pytest.fixture
def test_csv_path(tmp_path: Path) -> Path:
    csv_path: Path = tmp_path / 'test.csv'
    df = pl.DataFrame({
        'x': [float(x % 2) for x in range(1, 1000)],
        'y': [float(y % 7) for y in range(1, 1000)],
        'z': [float(z % 5) for z in range(1, 1000)],
        'target': [float(t % 10) for t in range(1, 1000)],
    })
    df.write_csv(csv_path)
    return csv_path

def test_endtoend_low_level(test_csv_path: Path, ddb_manager: DuckdbManager) -> None:
    # This is an example of the lower-level, explicit API, where ingest into duckdb is a separate step
    # and the user needs to manage the names of the databases and tables, whether to overwrite data if it's already there, etc
    # (we still have a TODO to separate context gen from the feature search runner into the ingest, and to 
    #  give the ingest an explicit runner interface)

    # Separate function to clearly separate concerns
    def ingest_data() -> FeatureSearchInputData:
        with ddb_manager.cursor() as conn:
            csv_input: DatasetSource = DatasetSource.from_csv(test_csv_path, conn)
            table_name = DuckdbName.qualify(test_csv_path.name, conn)
            DuckdbTableSink(table_name).write(csv_input, conn)
            table = DuckdbTable.from_duckdb(table_name, conn)
            split = sampling.split_duckdb_table(conn, table.name)
            with conn.cursor() as cursor:
                _logger.info(f'Split table: {cursor.sql(f'SELECT * FROM {table.name}')}')
            input_data = FeatureSearchInputData.from_split_table(split, 'target', TablesWithContextDefinitions.from_list([]), conn)
            return input_data

    # Not attaching any on-disk databases
    run_context = RunContext.create_default_context(ddb_manager)

    input_data = ingest_data()

    # Multiple generators, select based on stats
    input_params1 = RegressionFeatureSearchParams( # TODO also test classification
        generators=(ToySyncFeatureGenerator(), ToyAsyncFeatureGenerator()),
        selector=ToySyncFeatureSelector()
    )
    results1 = SimpleFeatureSearchRunner().run(run_context, input_data, input_params1)
    _logger.info(results1)
    # TODO assert stuff

    # Single generator because we haven't implemented feature name deduplication yet, and EnrichedFeatureSelector
    input_params2 = RegressionFeatureSearchParams( # TODO also test classification
        generators=(ToySyncFeatureGenerator(), ),
        selector=ToyAsyncEnrichedFeatureSelector()
    )
    results2 = SimpleFeatureSearchRunner().run(run_context, input_data, input_params2)
    _logger.info(results2)
    # TODO assert stuff

