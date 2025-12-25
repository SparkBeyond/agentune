from collections.abc import Sequence

import attrs
from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.feature.base import Feature
from agentune.analyze.feature.sql.base import FeatureFromQueryCtor, SqlFeatureCorrector
from agentune.analyze.feature.validate.base import (
    FeatureValidationCode,
    FeatureValidationError,
    FeatureValidator,
)
from agentune.core.database import DuckdbTable
from agentune.core.dataset import Dataset
from agentune.core.schema import Schema


@frozen
class ValidateAndRetryParams:
    """The arguments to validate_and_retry. See the docs of validate_and_retry and feature_from_query.

    Writing them as a class makes implementing recursive functions simpler.
    """
    feature_ctor: FeatureFromQueryCtor

    # Mandatory parameters to FeatureFromQueryCtor
    conn: DuckDBPyConnection
    sql_query: str
    params: Schema
    secondary_tables: Sequence[DuckdbTable]

    # Parameters to validate_and_retry
    input: Dataset
    max_global_retries: int
    max_local_retries: int
    corrector: SqlFeatureCorrector
    validators: Sequence[FeatureValidator]

    # Optional parameters to FeatureFromQueryCtor

    primary_table_name: str = 'primary_table'
    index_column_name: str = 'rowid'

    name: str | None = None
    description: str = ''
    technical_description: str | None = None


async def validate_and_retry(params: ValidateAndRetryParams) -> Feature | None:
    """Validate the feature and try to fix it if it's invalid.

    Try each validator in sequence on the feature. The feature_ctor acts as the first validator.
    If it fails, call the corrector to try to fix the feature and restart.
    If all validators pass, return the (possibly corrected) feature; if we run out of budget or if the corrector ever
    returns None (i.e. gives up), return None.

    Args:
        max_global_retries: how many times in total we can call the Corrector before giving up
        max_local_retries: how many times we can call the corrector for the same problem in sequence.
                           'The same problem' is defined by FeatureValidationError.code.
                           This only applies in sequence; it doesn't prevent us from trying to correct the errors
                           A, B, A, B, ...

    The other parameters are passed to the ctor; see the documentation of feature_from_query.
    """
    return await _validate_and_retry(params, None, params.max_global_retries, params.max_local_retries)

async def _validate_and_retry(params: ValidateAndRetryParams,
                              current_error_code: FeatureValidationCode | None,
                              remaining_global_retries: int,
                              remaining_local_retries: int) -> Feature | None:
    try:
        feature = params.feature_ctor(params.conn, params.sql_query, params.params, params.secondary_tables, params.primary_table_name,
                                      params.index_column_name, params.name, params.description, params.technical_description)
    except FeatureValidationError as error:
        return await _handle_error(params, current_error_code, remaining_global_retries, remaining_local_retries, error)

    for validator in params.validators:
        try:
            await validator.validate(feature, params.input, params.conn)
        except FeatureValidationError as error:
            return await _handle_error(params, current_error_code, remaining_global_retries, remaining_local_retries, error)

    return feature

async def _handle_error(params: ValidateAndRetryParams,
                        current_error_code: FeatureValidationCode | None,
                        remaining_global_retries: int,
                        remaining_local_retries: int,
                        error: FeatureValidationError) -> Feature | None:
    if remaining_global_retries == 0:
        return None

    if error.code != current_error_code:
        remaining_local_retries = params.max_local_retries
    if remaining_local_retries == 0:
        return None

    new_query = await params.corrector.correct(params.sql_query, error)
    if new_query is None:
        return None
    return await _validate_and_retry(
        attrs.evolve(params, sql_query=new_query),
        error.code, remaining_global_retries - 1, remaining_local_retries - 1
    )
