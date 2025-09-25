import logging

import attrs
import polars as pl
import pytest
from duckdb.duckdb import DuckDBPyConnection

from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbName, DuckdbTable
from agentune.analyze.core.schema import Field
from agentune.analyze.feature.problem import (
    ClassificationProblem,
    ProblemDescription,
    RegressionDirection,
    RegressionProblem,
    TableDescription,
)
from agentune.analyze.join.base import TablesWithJoinStrategies, TableWithJoinStrategies
from agentune.analyze.run.base import RunContext
from agentune.analyze.run.feature_search import problem_discovery

from .test_impl import input_data_from_df

_logger = logging.getLogger(__name__)

def test_discover_problem_int_target(run_context: RunContext, conn: DuckDBPyConnection) -> None:
    input_data = input_data_from_df(run_context, pl.DataFrame({
        'x': [float(x % 10) for x in range(1, 1000)],
        'target': [int(t % 3) for t in range(1, 1000)],
    }))

    def dotest(description: ProblemDescription, expected_classification: bool, max_classes: int = 20) -> None:
        problem = problem_discovery.discover_problem(input_data, description, conn, max_classes)
        problem_discovery.validate_input(input_data, problem, conn)
        if expected_classification:
            assert problem == ClassificationProblem(description, Field('target', types.int64), (0, 1, 2))
        else:
            assert problem == RegressionProblem(description, Field('target', types.int64))

    dotest(ProblemDescription('target'), True)
    dotest(ProblemDescription('target', problem_type='classification'), True)
    dotest(ProblemDescription('target', problem_type='regression'), False)
    dotest(ProblemDescription('target', target_desired_outcome=RegressionDirection.up), False)
    dotest(ProblemDescription('target', target_desired_outcome=1), True)

    with pytest.raises(ValueError, match='not in list of classes'):
        dotest(ProblemDescription('target', target_desired_outcome=3), True)
    with pytest.raises(ValueError, match='not in list of classes'):
        dotest(ProblemDescription('target', target_desired_outcome='1'), True)
    with pytest.raises(ValueError, match='more than 2 classes'):
        dotest(ProblemDescription('target'), True, 2)


def test_discover_problem_float_target(run_context: RunContext, conn: DuckDBPyConnection) -> None:
    input_data = input_data_from_df(run_context, pl.DataFrame({
        'x': [float(x % 10) for x in range(1, 1000)],
        'target': [float(t % 3 + 0.1) for t in range(1, 1000)],
    }))

    def dotest(description: ProblemDescription) -> None:
        problem = problem_discovery.discover_problem(input_data, description, conn, 20)
        problem_discovery.validate_input(input_data, problem, conn)
        assert problem == RegressionProblem(description, Field('target', types.float64))

    dotest(ProblemDescription('target'))
    dotest(ProblemDescription('target', problem_type='regression'))
    dotest(ProblemDescription('target', target_desired_outcome=RegressionDirection.up))

    with pytest.raises(ValueError, match='not supported for classification'):
        dotest(ProblemDescription('target', problem_type='classification'))
    with pytest.raises(ValueError, match='not supported with desired outcome 1'):
        dotest(ProblemDescription('target', target_desired_outcome=1))


def test_discover_problem_str_target(run_context: RunContext, conn: DuckDBPyConnection) -> None:
    # Use non-numeric string classes to ensure validation doesn't rely on float casting
    labels = ('A', 'B', 'C')
    num_labels = len(labels)
    input_data = input_data_from_df(run_context, pl.DataFrame({
        'x': [float(x % 10) for x in range(1, 1000)],
        'target': [labels[t % num_labels] for t in range(1, 1000)],
    }))

    def dotest(description: ProblemDescription, max_classes: int = 20) -> None:
        problem = problem_discovery.discover_problem(input_data, description, conn, max_classes)
        problem_discovery.validate_input(input_data, problem, conn)
        assert problem == ClassificationProblem(description, Field('target', types.string), labels)

    dotest(ProblemDescription('target'))
    dotest(ProblemDescription('target', problem_type='classification'))
    dotest(ProblemDescription('target', target_desired_outcome='B'))

    with pytest.raises(ValueError, match='regression target must be numeric'):
        dotest(ProblemDescription('target', problem_type='regression'))
    with pytest.raises(ValueError, match='dtype str not supported with desired outcome up'):
        dotest(ProblemDescription('target', target_desired_outcome=RegressionDirection.up))


def test_target_dtype_preserved(run_context: RunContext, conn: DuckDBPyConnection) -> None:
    input_data = input_data_from_df(run_context, pl.DataFrame({
        'x': [float(x % 10) for x in range(1, 1000)],
        'target': [int(t % 3) for t in range(1, 1000)],
    }, schema_overrides={'target': types.int16.polars_type}))
    problem = problem_discovery.discover_problem(input_data, ProblemDescription('target'), conn, 20)
    assert problem.target_column.dtype == types.int16, 'Correct target column dtype'


def test_fail_on_invalid_int_target_values(run_context: RunContext) -> None:
    with run_context.ddb_manager.cursor() as conn:
        input_data = input_data_from_df(run_context, pl.DataFrame({
            'x': [float(x % 10) for x in range(1, 1000)],
            'target': [int(t % 3) for t in range(1, 1000)],
        }))
        problem = RegressionProblem(
            ProblemDescription('target'),
            input_data.feature_search.schema['target']
        )

        # Passes without missing values
        problem_discovery.validate_input(input_data, problem, conn)

        input_data_missing_search = attrs.evolve(input_data,
                                                 feature_search=attrs.evolve(input_data.feature_search,
                                                                             data=input_data.feature_search.data.vstack(pl.DataFrame({
                                                                                 'x': [1.0],
                                                                                 'target': [None]
                                                                             }))))
        with pytest.raises(ValueError, match=r'missing values.*feature search dataset'):
            problem_discovery.validate_input(input_data_missing_search, problem, conn)

        conn.execute('insert into input(x, target, _is_train, _is_feature_search, _is_feature_eval) values (1.0, null, true, false, true)')
        with pytest.raises(ValueError, match=r'missing values.*feature evaluation dataset'):
            problem_discovery.validate_input(input_data, problem, conn)

        conn.execute('update input set _is_feature_eval=false where target is null')
        with pytest.raises(ValueError, match=r'missing values.*train dataset'):
            problem_discovery.validate_input(input_data, problem, conn)

        conn.execute('update input set _is_train=false where target is null')
        with pytest.raises(ValueError, match=r'missing values.*test dataset'):
            problem_discovery.validate_input(input_data, problem, conn)


def test_fail_on_invalid_float_target_values(run_context: RunContext) -> None:
    with run_context.ddb_manager.cursor() as conn:
        input_data = input_data_from_df(run_context, pl.DataFrame({
            'x': [float(x % 10) for x in range(1, 1000)],
            'target': [float(t % 3) for t in range(1, 1000)],
        }))
        problem = RegressionProblem(
            ProblemDescription('target'),
            input_data.feature_search.schema['target']
        )
        
        # Passes without missing values
        problem_discovery.validate_input(input_data, problem, conn)

        input_data_missing_search = attrs.evolve(input_data,
                                                 feature_search=attrs.evolve(input_data.feature_search,
                                                                             data=input_data.feature_search.data.vstack(pl.DataFrame({
                                                                                 'x': [1.0],
                                                                                 'target': [None]
                                                                             }))))
        with pytest.raises(ValueError, match=r'missing values.*feature search dataset'):
            problem_discovery.validate_input(input_data_missing_search, problem, conn)

        for invalid_value in ['null', "'nan'::float", "'inf'::float", "'-inf'::float"]:
            conn.execute(f'insert into input(x, target, _is_train, _is_feature_search, _is_feature_eval) values (1.0, {invalid_value}, true, false, true)')
            with pytest.raises(ValueError, match=r'missing values.*feature evaluation dataset'):
                problem_discovery.validate_input(input_data, problem, conn)

            operator = 'is' if invalid_value == 'null' else '=='

            conn.execute(f'update input set _is_feature_eval=false where target {operator} {invalid_value}')
            with pytest.raises(ValueError, match=r'missing values.*train dataset'):
                problem_discovery.validate_input(input_data, problem, conn)

            conn.execute(f'update input set _is_train=false where target {operator} {invalid_value}')
            with pytest.raises(ValueError, match=r'missing values.*test dataset'):
                problem_discovery.validate_input(input_data, problem, conn)

            # Clean for next cycle
            conn.execute(f'delete from input where target {operator} {invalid_value}')
            assert conn.fetchone() == (1,)

def test_table_descriptions(run_context: RunContext, conn: DuckDBPyConnection) -> None:
    conn.execute('create table secondary(y int)')
    secondary_table = DuckdbTable.from_duckdb('secondary', conn)

    input_data = input_data_from_df(run_context, pl.DataFrame({
        'x': [float(x % 10) for x in range(1, 1000)],
        'target': [float(t % 3) for t in range(1, 1000)],
    }))
    input_data = attrs.evolve(input_data, join_strategies=TablesWithJoinStrategies.from_list([
        TableWithJoinStrategies(secondary_table, {})
    ]))

    problem = RegressionProblem(
        ProblemDescription('target'),
        input_data.feature_search.schema['target']
    )
    problem_discovery.validate_input(input_data, problem, conn) # No descriptions - OK

    problem = RegressionProblem(
        ProblemDescription('target', main_table=TableDescription(column_descriptions={'x': 'x column', 'target': 'target column'})),
        input_data.feature_search.schema['target']
    )
    problem_discovery.validate_input(input_data, problem, conn) # Description of main table column - OK

    with pytest.raises(ValueError, match='column y but it does not exist in feature search input'):
        problem = RegressionProblem(
            ProblemDescription('target', main_table=TableDescription(column_descriptions={'y': 'y column'})),
            input_data.feature_search.schema['target']
        )
        problem_discovery.validate_input(input_data, problem, conn)

    problem = RegressionProblem(
        ProblemDescription('target', secondary_tables={secondary_table.name: TableDescription('secondary table')}),
        input_data.feature_search.schema['target']
    )
    problem_discovery.validate_input(input_data, problem, conn) # Description of secondary table - OK

    problem = RegressionProblem(
        ProblemDescription('target', secondary_tables={secondary_table.name: TableDescription('secondary table', column_descriptions={'y': 'y column'})}),
        input_data.feature_search.schema['target']
    )
    problem_discovery.validate_input(input_data, problem, conn) # Description of secondary table column - OK

    with pytest.raises(ValueError, match='it does not exist in feature search input'):
        problem = RegressionProblem(
            ProblemDescription('target', secondary_tables={DuckdbName.qualify('tertiary', conn): TableDescription('secondary table', column_descriptions={'y': 'y column'})}),
            input_data.feature_search.schema['target']
        )
        problem_discovery.validate_input(input_data, problem, conn)

    with pytest.raises(ValueError, match='column z but it does not exist in table'):
        problem = RegressionProblem(
            ProblemDescription('target', secondary_tables={secondary_table.name: TableDescription('secondary table', column_descriptions={'z': 'z column'})}),
            input_data.feature_search.schema['target']
        )
        problem_discovery.validate_input(input_data, problem, conn)
