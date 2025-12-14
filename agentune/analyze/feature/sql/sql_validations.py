# Validators specific to SqlBackedFeature
import datetime

import attrs
from _duckdb import DuckDBPyConnection

from agentune.analyze.feature.sql.base import SqlBackedFeature
from agentune.analyze.feature.validate.base import FeatureValidator
from agentune.core.dataset import Dataset


@attrs.frozen
class TimeoutValidator(FeatureValidator[SqlBackedFeature]):
    """Checks that the feature doesn't time out when evaluated on this input.

    The timeout implementation is in SqlBackedFeature itself (including raising an appropriate FeatureValidationError).
    Using this class is equivalent to creating the feature with the appropriate timeout parameter and passing it
    to other validators, and so can be skipped.
    """

    timeout: datetime.timedelta

    async def validate(self, feature: SqlBackedFeature, input: Dataset, conn: DuckDBPyConnection) -> None:
        feature = attrs.evolve(feature, timeout = self.timeout)
        await feature.acompute_batch_safe(input, conn)
