from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Literal, cast, override

from attrs import field, frozen
from frozendict import frozendict

from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbName
from agentune.analyze.core.schema import Field
from agentune.analyze.core.types import Dtype
from agentune.analyze.util.attrutil import frozendict_converter
from agentune.analyze.util.cattrutil import UseTypeTag

type TargetKind = Literal['classification', 'regression']
type Classification = Literal['classification']
type Regression = Literal['regression']

type ClassificationClass = bool | int | str
'''A class in a classification problem.

The dtype of the target column can be narrower, e.g. uint16 or enum; this is the type of the target values
when they appear as scalars in Python code. 
'''

class RegressionDirection(StrEnum):
    up = 'up'
    down = 'down'

@frozen
class TableDescription:
    """A natural-language description of an input data table.

    Args:
        description: the entity corresponding to each row in the table (e.g. 'kittens') and any additional
                     information on that entity.
        column_descriptions: descriptions of the columns in the table, mapped by the column name.
                             E.g., {'age': 'age of the kitten in weeks', 'color': 'color of the kitten'}
    """
    description: str | None = None
    column_descriptions: frozendict[str, str] = field(default=frozendict(), converter=frozendict_converter)

@frozen
class ProblemDescription:
    """User input to the analyzer. Almost all parameters are optional.

    During analysis, an instance of class Problem is created which contains a copy of this class; that copy
    might be modified from the original user input by setting some fields that the user left empty.

    Parameters which describe data (schema and/or values) are validated against the data inputs to `analyze`.

    Args:
        target_column: name of the column in the main table that contains the target variable to predict/estimate/explain.
                       The target variable is the manifestation or indication of the outcome or indicator associated with
                       the corresponding row. Aggregated over the main table, it provides the KPI to be optimized.
        problem_type: 'classification' or 'regression'. If not specified, the type is inferred from the target column dtype.
                      Regression requires a numeric target column, and is the inferred type for numeric targets.
                      Classification allows all types (including numeric ones) and requires a limited number of distinct
                      values in the target column in the train dataset; the limit is given by `AnalyzeParams.max_classes`.
                      If this is None, the actual (inferred) problem_type is given by `Problem.target_kind` when analysis
                      completes.
        target_desired_outcome: the preferred outcome for the target variable. This causes features to be found
                                that drive this outcome in particular, and recommendations to be made that optimize for
                                that outcome.
                                For classification problems, this is a target value, which must exactly match one of the
                                target values present in the train data. For regression problems, this indicates whether
                                higher or lower target values are better (`RegressionDirection.up` or `RegressionDirection.down`).
        name:         a short name for the problem being analyzed, e.g. 'Kitten adoption'.
        description:  a longer (1-2 sentences) description, e.g. 'There are kittens in shelters that need to be adopted
                      to loving homes. This requires matching each kitten with the right person.'
        target_description: any additional information on the target and the possible outcomes, e.g.
                      'how long the kitten spent in the shelter before being adopted'.
        business_domain: the domain to which the problem and the data belong, e.g. 'animal welfare'.
        date_column:  Name of a column of type datetime in the main table.
                      Relevant for temporal problems. In the future, this will help determine the train-test split
                      to avoid target leaks.
        comments:     Any additional semantic information that can help analyze the problem and suggest improvements,
                      e.g. 'Only weaned kittens are put out for adoption'.
        main_table:   A description of the main table and its columns.
        secondary_tables: A mapping from table names to descriptions of secondary tables.
    """
    target_column: str # Comes first because it's the only mandatory parameter
    problem_type: TargetKind | None = None
    target_desired_outcome: ClassificationClass | RegressionDirection | None = None
    name: str | None = None
    description: str | None = None
    target_description: str | None = None
    business_domain: str | None = None
    date_column: str | None = None
    comments: str | None = None
    main_table: TableDescription | None = None # To comment on other columns of the main table
    secondary_tables: frozendict[DuckdbName, TableDescription] = field(default=frozendict(), converter=frozendict_converter)

    def __attrs_post_init__(self) -> None:
        if self.problem_type == 'classification' and isinstance(self.target_desired_outcome, RegressionDirection):
            raise ValueError('RegressionDirection cannot be used with classification problem type.')
        if self.problem_type == 'regression' and \
                self.target_desired_outcome is not None and not isinstance(self.target_desired_outcome, RegressionDirection):
            raise ValueError('Desired outcome class cannot be used with regression problem type.')


@frozen
class Problem(ABC, UseTypeTag):
    """Final information about the problem.

    The ProblemDescription may not be the original one provided by the user; some attributes left empty in the user input
    may have been filled in automatically.

    The attributes of this class and of the included ProblemDescription (if set) are guaranteed to be
    valid and consistent with each other and the data.

    Some attributes of this class match attributes of ProblemDescription but have types that provide more information;
    for example, target_column is a Field instead of a string. Code should use attributes in this class in preference
    to attributes of the contained ProblemDescription.
    """
    problem_description: ProblemDescription
    target_column: Field
    date_column: Field | None = None

    def __attrs_post_init__(self) -> None:
        if self.date_column is not None and not self.date_column.dtype.is_temporal():
            raise ValueError(f'Date column must have temporal type (date, time or timestamp) and not {self.date_column.dtype}')

        if self.target_column.name != self.problem_description.target_column:
            raise ValueError(f'Mismatch with problem_description: target column {self.problem_description.target_column} vs {self.target_column.name}')
        if self.problem_description.date_column is not None and \
                (self.date_column is None or self.date_column.name != self.problem_description.date_column):
            raise ValueError(f'Mismatch with problem_description: date column {self.problem_description.date_column} vs {self.date_column}')

    @property
    @abstractmethod
    def target_kind(self) -> TargetKind: ...

@frozen
class ClassificationProblem(Problem):
    classes: tuple[ClassificationClass, ...]
    date_column: Field | None = None # Redeclare optional parameters to put them after mandatory parameters

    @property
    def target_desired_outcome(self) -> ClassificationClass | None:
        """Same value as self.problem_description.target_desired_outcome but narrower type, guaranteed by ctor check."""
        # This should need a cast(), but mypy doesn't want it and complains if it's present
        return self.problem_description.target_desired_outcome

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()

        if self.classes != tuple(sorted(self.classes)):
            raise ValueError('Classes must be canonically ordered (i.e. sorted in ascending order)')

        if self.problem_description.problem_type == 'regression':
            raise ValueError('Mismatch with problem_description: problem type')
        if isinstance(self.problem_description.target_desired_outcome, RegressionDirection):
            raise ValueError('RegressionDirection.up/down cannot be used with classification problem')

        if len(self.classes) < 2: # noqa: PLR2004
            raise ValueError('Classification problem must have at least 2 classes.')
        if self.target_desired_outcome is not None and self.target_desired_outcome not in self.classes:
            raise ValueError(f'Desired outcome class {self.target_desired_outcome} not in list of classes: {self.classes}')

        # Same as python_type_from_duckdb for the dtypes we allow here
        expected_python_type = types.python_type_from_polars(self.target_column.dtype)
        if not all(isinstance(value, expected_python_type) for value in self.classes):
            raise ValueError(f"Types of classes {self.classes} don't match target column dtype {self.target_column.dtype}, "
                             f"expected values of type {expected_python_type} but found {', '.join(str(type(cls)) for cls in self.classes)}")

        match self.target_column.dtype:
            case types.EnumDtype(values):
                if not all(cast(str, value) in values for value in self.classes):
                    raise ValueError(f"List of classes {self.classes} doesn't match target column enum type's list of values: {values}")
            case dtype if self.is_allowed_dtype(dtype): pass
            case other: raise ValueError(f'Dtype {other} not allowed as classification target type. '
                                         f'Allowed types are int (of any size and signedness), bool, string, enum.')

    @override
    @property
    def target_kind(self) -> TargetKind:
        return 'classification'

    @staticmethod
    def is_allowed_dtype(dtype: Dtype) -> bool:
        """Is this dtype allowed for a classification target"""
        return dtype.is_integer() or dtype in (types.string, types.boolean) or isinstance(dtype, types.EnumDtype)


@frozen
class RegressionProblem(Problem):

    @property
    def target_desired_outcome(self) -> RegressionDirection | None:
        """Same value as self.problem_description.target_desired_outcome but narrower type, guaranteed by ctor check."""
        return cast(RegressionDirection | None, self.problem_description.target_desired_outcome)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()

        if self.problem_description.problem_type == 'classification':
            raise ValueError('Mismatch with problem_description: problem type')
        if self.problem_description.target_desired_outcome is not None and \
                not isinstance(self.problem_description.target_desired_outcome, RegressionDirection):
            raise ValueError('target_desired_outcome class value cannot be used with regression problem')

        if not self.target_column.dtype.is_numeric():
            raise ValueError(f'Target column dtype {self.target_column.dtype} must be numeric for regression problem.')

    @override
    @property
    def target_kind(self) -> TargetKind:
        return 'regression'
