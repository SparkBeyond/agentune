import logging
from typing import cast

from duckdb import DuckDBPyConnection, DuckDBPyRelation

from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.schema import Schema
from agentune.analyze.feature.problem import (
    ClassificationClass,
    ClassificationProblem,
    Problem,
    ProblemDescription,
    RegressionDirection,
    RegressionProblem,
    TableDescription,
    TargetKind,
)
from agentune.analyze.run.analysis.base import AnalyzeInputData

_logger = logging.getLogger(__name__)


def discover_problem(data: AnalyzeInputData, description: ProblemDescription,
                     conn: DuckDBPyConnection, max_classes: int,
                     target_kind_override: TargetKind | None = None) -> ClassificationProblem | RegressionProblem:
    """Determine the problem parameters: target kind, list of classes, etc.

    Note that the concrete classes that can be returned are ClassificationProblem and RegressionProblem;
    the base class Problem is abstract.

    The classes are discovered from the train dataset, which is read once.
    This can take unbounded time, so it should be run on the threadpool.

    It can also run out of memory or temp disk space if the target column has very high cardinality.
    We could try to protect against that, but it would either require reading the input twice,
    or reading the input column into python and computing distinct values ourselves which would be much slower.

    Args:
        max_classes: when discovering classes, if more than this number of classes are found, fail with error.
                     Ignored in regression mode.
    """
    if description.target_column not in data.feature_search.schema.names:
        raise ValueError(f'Target column {description.target_column} not found')
    target_column = data.feature_search.schema[description.target_column]
    target_dtype = target_column.dtype

    if description.date_column is not None:
        if description.date_column not in data.feature_search.schema.names:
            raise ValueError(f'Date column {description.date_column} not found')
        date_column = data.feature_search.schema[description.date_column]
        if not date_column.dtype.is_temporal():
            raise ValueError(f'Date column {date_column} must have a temporal dtype (date, time or timestamp), found {date_column.dtype}')
    else:
        date_column = None

    # Some validation (on matching the ProblemDescription) is performed inside Problem subclass constructors,
    # e.g. that the preferred class is in the classes list

    target_kind: TargetKind | None = target_kind_override or description.problem_type
    match target_kind:
        case 'classification':
            if not ClassificationProblem.is_allowed_dtype(target_dtype):
                raise ValueError(f'Target column {description.target_column} dtype {target_dtype} is not supported for classification. '
                                 f'Supported dtypes are integers, bool, string and enum.')

            classes: tuple[ClassificationClass, ...]
            if isinstance(target_dtype, types.EnumDtype):
                classes = target_dtype.values
            else:
                with conn.cursor() as cursor:
                    source_rel = data.train.to_duckdb(cursor)
                    cursor.register('source', source_rel)
                    # Unfortunately, the `limit` doesn't stop duckdb from collecting all distinct values in memory first,
                    # even if there are many more than the limit.
                    cursor.execute(f'''select distinct "{target_column.name}" from source limit $1''', [max_classes + 1])
                    classes = cast(tuple[ClassificationClass, ...], # True by construction, we checked the dtype
                                   tuple(result[0] for result in cursor.fetchmany(max_classes + 1)))
            if len(classes) > max_classes:
                raise ValueError(f'Target column has more than {max_classes} classes')
            if len(classes) < 2: # noqa: PLR2004
                raise ValueError(f'Target column has fewer than 2 distinct values: {', '.join(str(cls) for cls in classes)}')
            classes = tuple(sorted(classes))
            return ClassificationProblem(description, target_column, classes, date_column)

        case 'regression':
            if not target_dtype.is_numeric():
                raise ValueError(f'A regression target must be numeric, but {description.target_column} has dtype {target_dtype}')
            return RegressionProblem(description, target_column, date_column)

        case None:
            if ClassificationProblem.is_allowed_dtype(target_dtype) and not isinstance(description.target_desired_outcome, RegressionDirection):
                return discover_problem(data, description, conn, max_classes, target_kind_override='classification')
            elif target_dtype.is_numeric() and (isinstance(description.target_desired_outcome, RegressionDirection) or description.target_desired_outcome is None):
                return discover_problem(data, description, conn, max_classes, target_kind_override='regression')
            else:
                raise ValueError(f'Target column {description.target_column} dtype {target_dtype} not supported '
                                f'with desired outcome {description.target_desired_outcome}')

def _validate_table_description(description: TableDescription, schema: Schema, name: str) -> None:
    for col_name in description.column_descriptions:
        if col_name not in schema.names:
            raise ValueError(f'Description given for column {col_name} but it does not exist in {name}')

def _validate_secondary_table(table: DuckdbTable, conn: DuckDBPyConnection, description: ProblemDescription) -> None:
    if not DuckdbTable.exists(table.name, conn):
        raise ValueError(f'Secondary table {table.name} does not exist')
    actual_table = DuckdbTable.from_duckdb(table.name, conn)
    if actual_table.schema != table.schema:
        raise ValueError(f'Secondary table {table.name} has schema {actual_table.schema} but input specifies schema {table.schema}')
    if actual_table.name in description.secondary_tables:
        _validate_table_description(description.secondary_tables[actual_table.name], actual_table.schema, f'table {table.name}')



def validate_input(data: AnalyzeInputData, problem: Problem, conn: DuckDBPyConnection) -> None:
    """Fail if the target column in any input dataset has missing values,
       or non-finite values if it is a float column.

    This requires reading the input datasets an extra time, which can be expensive
    if they are not stored in duckdb. In particular, without this, we could guarantee
    only reading the test dataset once (streaming it), and probably the full train dataset too.

    A future improvement can move the check to be done while streaming the dataset,
    but for now the decision was to check ahead of time.

    This can take unbounded time, so it should be run on the threadpool.
    """
    if problem.problem_description.main_table is not None:
        _validate_table_description(problem.problem_description.main_table, data.feature_search.schema, 'analysis input')
    for secondary_table in problem.problem_description.secondary_tables:
        if secondary_table not in data.join_strategies.tables:
            raise ValueError(f'Description given for secondary table {secondary_table} but it does not exist in analysis input')
    for secondary_table_with_join_strategies in data.join_strategies.tables.values():
        _validate_secondary_table(secondary_table_with_join_strategies.table, conn, problem.problem_description)

    target_name = problem.target_column.name
    target_is_float = problem.target_column.dtype.is_float()

    def _raise_if_not_empty(rel: DuckDBPyRelation, where_sql: str, message: str) -> None:
        """Raise ValueError(message) if any row matches where_sql in rel.

        Ensures we handle fetchone() typing safely and avoids duplicating logic.
        """
        count = rel.filter(where_sql).aggregate('count(*)').fetchone()
        match count:
            case (int(c), ) if c > 0:
                raise ValueError(message)
            case _:
                pass

    for name, source in [
        ('feature search', data.feature_search.as_source()),
        ('feature evaluation', data.feature_eval),
        ('train', data.train),
        ('test', data.test),
    ]:
        source_rel = source.to_duckdb(conn)
        if target_is_float:
            _raise_if_not_empty(
                source_rel,
                f'''"{target_name}" is null or "{target_name}" in ('nan'::float, 'inf'::float, '-inf'::float)''',
                f'Target column may not contain missing values or non-finite float values ({name} dataset)',
            )
        else:
            # For non-floating targets (ints, strings, enums, bool), only nulls are invalid.
            _raise_if_not_empty(
                source_rel,
                f'"{target_name}" is null',
                f'Target column may not contain missing values ({name} dataset)',
            )
