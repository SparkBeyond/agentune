from abc import ABC, abstractmethod
from enum import StrEnum

from duckdb import DuckDBPyConnection

from agentune.analyze.feature.base import Feature
from agentune.core.dataset import Dataset


class FeatureValidationCode(StrEnum):
    """Superclass of per-validator enums that document error codes contained by their FeatureValidationErrors."""

class FeatureValidationError(Exception):
    """An error in a feature's behavior flagged by a FeatureValidator.

    The `code` is a short string that is part of the validator's API and can be matched by other code to distinguish error types.
    """

    def __init__(self, code: FeatureValidationCode, message: str) -> None:
        super().__init__(code, message)
        self.code = code
        self.message = message

class FeatureValidator(ABC):
    @abstractmethod
    async def validate(self, feature: Feature, input: Dataset, conn: DuckDBPyConnection) -> None:
        """Raise a FeatureValidationError if the feature is invalid.

        If an error of any other type is raised, it is either raised by the feature itself or is a bug in the validator.

        The `input` and `conn` parameters should be valid for this feature's .compute method, i.e. they should have
        the right schema and tables defined. (Just as when calling feature.compute, they are allowed to have a schema
        with more columns, tables with extra columns, and extra tables.)

        The validator MAY use the input data, and it MAY create new data (or modify this data) in order to test the feature.

        The validator does not make any persistent changes to the connection, but it also cannot prevent the feature
        from making such changes until we implement read-only connections.

        A validator MAY decide it is inapplicable to this feature and return without an error.
        """
        ...

