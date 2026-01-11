import logging
from typing import override

from attrs import define, field
from duckdb import DuckDBPyConnection

from agentune.analyze.feature.sql.base import (
    SqlBackedFeature,
    SqlFeatureCorrector,
    SqlFeatureSpec,
)
from agentune.analyze.feature.sql.create import (
    QueryValidationCode,
    feature_from_query,
    float_feature_from_query,
    int_feature_from_query,
)
from agentune.analyze.feature.sql.validator_loop import ValidateAndRetryParams, validate_and_retry
from agentune.analyze.feature.validate.base import FeatureValidationError
from agentune.analyze.feature.validate.law_and_order import LawAndOrderValidator
from agentune.core.database import DuckdbTable

_logger = logging.getLogger(__name__)

@define
class TestFeatureCorrector(SqlFeatureCorrector):
    """Records the calls made to it in self.calls and responds from self.corrections until it runs out."""

    calls: list[tuple[str, FeatureValidationError]] = field(factory=list, init=False)
    corrections: list[str | None] = field(factory=list)

    @override
    async def correct(self, sql_feature_spec: SqlFeatureSpec, error: FeatureValidationError) -> SqlFeatureSpec | None:
        self.calls.append((sql_feature_spec.sql_query, error))
        if self.corrections:
            correction = self.corrections.pop(0)
            return SqlFeatureSpec(sql_query=correction) if correction is not None else None
        else:
            return sql_feature_spec


async def test_validator_loop(conn: DuckDBPyConnection) -> None:
    conn.execute('CREATE TABLE table1 as select unnest::int32 as value from unnest(range(100))')

    table1 = DuckdbTable.from_duckdb('table1', conn)
    input = table1.as_source().to_dataset(conn)
    validators = [LawAndOrderValidator()]

    feature = int_feature_from_query(conn,
                                     SqlFeatureSpec(sql_query='select value from primary_table'),
                                     table1.schema.drop('key'),
                                     [])

    # Validation passes with zero corrections
    corrector = TestFeatureCorrector()
    new_feature = await validate_and_retry(ValidateAndRetryParams(
        feature_from_query,
        conn, SqlFeatureSpec(sql_query=feature.sql_query), feature.params, feature.secondary_tables, input,
        1, 1,
        corrector, validators
    ))
    assert len(corrector.calls) == 0
    assert type(new_feature) is type(feature)
    assert isinstance(new_feature, SqlBackedFeature) and new_feature.sql_query == feature.sql_query

    # Validation passes with a zero budget if there are no errors
    await validate_and_retry(ValidateAndRetryParams(
        feature_from_query,
        conn, SqlFeatureSpec(sql_query=feature.sql_query), feature.params, feature.secondary_tables, input,
        0, 0,
        corrector, validators
    ))
    await validate_and_retry(ValidateAndRetryParams(
        feature_from_query,
        conn, SqlFeatureSpec(sql_query=feature.sql_query), feature.params, feature.secondary_tables, input,
        0, 1,
        corrector, validators
    ))
    await validate_and_retry(ValidateAndRetryParams(
        feature_from_query,
        conn, SqlFeatureSpec(sql_query=feature.sql_query), feature.params, feature.secondary_tables, input,
        1, 0,
        corrector, validators
    ))

    # Fail in ctor, succeed on second retry
    sql_query = 'select nonesuch from primary_table'
    corrector = TestFeatureCorrector(corrections = [sql_query, 'select value from primary_table'])
    new_feature = await validate_and_retry(ValidateAndRetryParams(
        feature_from_query,
        conn, SqlFeatureSpec(sql_query=sql_query), feature.params, feature.secondary_tables, input,
        2, 2,
        corrector, validators
    ))
    assert len(corrector.calls) == 2
    for query, error in corrector.calls:
        assert query == sql_query
        assert error.code == QueryValidationCode.illegal
    assert new_feature == feature

    # Fail in ctor, run out of local budget
    corrector = TestFeatureCorrector(corrections = [sql_query, 'select value from primary_table'])
    new_feature = await validate_and_retry(ValidateAndRetryParams(
        feature_from_query,
        conn, SqlFeatureSpec(sql_query=sql_query), feature.params, feature.secondary_tables, input,
        2, 1,
        corrector, validators
    ))
    assert new_feature is None
    assert len(corrector.calls) == 1

    # Fail in ctor, run out of global budget
    corrector = TestFeatureCorrector(corrections = [sql_query, 'select value from primary_table'])
    new_feature = await validate_and_retry(ValidateAndRetryParams(
        feature_from_query,
        conn, SqlFeatureSpec(sql_query=sql_query), feature.params, feature.secondary_tables, input,
        1, 2,
        corrector, validators
    ))
    assert new_feature is None
    assert len(corrector.calls) == 1

    # Face a different problem each time, eventually fix it, local budget does not constrain us
    sql_query = 'select nonesuch from primary_table'
    sql_query2 = 'select value from primary_table order by random()'
    corrector = TestFeatureCorrector(corrections = [sql_query2, sql_query, 'select value from primary_table'])
    new_feature = await validate_and_retry(ValidateAndRetryParams(
        feature_from_query,
        conn, SqlFeatureSpec(sql_query=sql_query), feature.params, feature.secondary_tables, input,
        10, 1,
        corrector, validators
    ))
    assert new_feature == feature
    assert len(corrector.calls) == 3

    # Face a different problem each time, eventually give up
    corrector = TestFeatureCorrector(corrections = [sql_query2, sql_query] * 10)
    new_feature = await validate_and_retry(ValidateAndRetryParams(
        feature_from_query,
        conn, SqlFeatureSpec(sql_query=sql_query), feature.params, feature.secondary_tables, input,
        10, 1,
        corrector, validators
    ))
    assert new_feature is None
    assert len(corrector.calls) == 10


async def test_validator_loop_specific_feature_type_ctor(conn: DuckDBPyConnection) -> None:
    conn.execute('CREATE TABLE table1 as select unnest::int32 as value from unnest(range(100))')

    table1 = DuckdbTable.from_duckdb('table1', conn)
    input = table1.as_source().to_dataset(conn)
    validators = [LawAndOrderValidator()]

    feature = feature_from_query(conn,
                                 SqlFeatureSpec(sql_query='select value from primary_table'),
                                 table1.schema.drop('key'),
                                 [])

    corrector = TestFeatureCorrector()
    # Specific type ctor
    new_feature = await validate_and_retry(ValidateAndRetryParams(
        int_feature_from_query,
        conn, SqlFeatureSpec(sql_query=feature.sql_query), feature.params, feature.secondary_tables, input,
        1, 1,
        corrector, validators
    ))
    assert type(new_feature) is type(feature)
    assert isinstance(new_feature, SqlBackedFeature) and new_feature.sql_query == feature.sql_query

    # Wrong type for query
    new_feature = await validate_and_retry(ValidateAndRetryParams(
        float_feature_from_query,
        conn, SqlFeatureSpec(sql_query=feature.sql_query), feature.params, feature.secondary_tables, input,
        1, 1,
        corrector, validators
    ))
    assert new_feature is None
