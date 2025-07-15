from collections.abc import Iterable
from typing import override

from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.context.base import KtsContext, KtsContextDefinition, TimeSeries, TimeWindow
from agentune.analyze.util import dataconv


@frozen
class KtsContextImpl[K](KtsContext[K]):
    conn: DuckDBPyConnection
    definition: KtsContextDefinition

    @override
    def get(self, key: K, window: TimeWindow, value_cols: Iterable[str]) -> TimeSeries | None: 
        start_op = '>=' if window.include_start else '>'
        end_op = '<=' if window.include_end else '<'
        sample_clause = f'USING SAMPLE {window.sample_maxsize} (reservoir, 42)' if window.sample_maxsize else ''
        
        # Sampling is not deterministic when multithreaded, even when a random seed is provided
        # And we can only set threads globally (per database), not locally (per connection).
        # Also I suspect that setting threads replaces the whole threadpool, which is expensive. 
        # So right now I'm leaving out the 'set threads', meaning this is not deterministic,
        # and we need TODO better.
        try:
            if window.sample_maxsize:
                #self.conn.execute('set threads = 1')
                pass
            # Join key_exists to get a single row of nulls if the key is not found
            # NOTE the USING SAMPLE clause applies to the table, after joins but before any WHERE filtering,
            #  so we need to use a join with a subquery instead of a simple filter on key and dates
            relation = self.conn.sql(f'''
                WITH 
                    key_exists AS (
                        SELECT exists(
                            SELECT 1 FROM "{self.relation_name}" WHERE "{self.definition.key_col.name}" = $key
                        ) AS key_exists
                    ),
                    main_table as (
                        SELECT "{self.definition.date_col.name}",
                                {", ".join(f'"{col}"' for col in value_cols)}
                        FROM "{self.relation_name}"
                        WHERE "{self.definition.key_col.name}" = $key
                            AND "{self.definition.date_col.name}" {start_op} $start
                            AND "{self.definition.date_col.name}" {end_op} $end
                        ORDER BY "{self.definition.date_col.name}"
                    )
                SELECT key_exists.key_exists, main_table.*
                FROM key_exists
                LEFT JOIN main_table on 1
                {sample_clause}
            ''', params={'key': key, 'start': window.start, 'end': window.end})
            
            # TODO output column key_exists could conflict with the name of another column;
            #  duckdb allows multiple columns with the same name in a result set, but polars doesn't
            dataset = dataconv.duckdb_to_dataset(relation)
            dataset_without_key_exists = dataset.drop('key_exists')

            # Handle special cases
            if dataset.data.height == 1:
                key_found = dataset.data['key_exists'].any()
                if key_found:
                    if dataset_without_key_exists.data[self.definition.date_col.name].is_null().all():
                        # Key found but no data in time range; return empty time series
                        return TimeSeries(dataset_without_key_exists.empty(), self.definition.date_col.name)
                else:
                    # key not found
                    return None        
           
            return TimeSeries(dataset_without_key_exists, self.definition.date_col.name)
        finally:
            if window.sample_maxsize:
                #self.conn.execute('reset threads')
                pass
