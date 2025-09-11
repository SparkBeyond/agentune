import datetime
import logging
from typing import Literal

import duckdb
import polars as pl

from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.dataset import duckdb_to_polars
from agentune.analyze.join.timeseries import KtsJoinStrategy, TimeWindow

_logger = logging.getLogger(__name__)

def test_timeseries() -> None:
    with duckdb.connect(':memory:lookup') as conn:
        conn.execute('create table test(key integer, date timestamp_ms, val1 integer, val2 varchar)')

        table = DuckdbTable.from_duckdb('test', conn)
        strategy = KtsJoinStrategy[int]('ts', table, table.schema['key'], table.schema['date'],
                                       (table.schema['val1'], table.schema['val2']))
        strategy.index.create(conn, 'test')

        included_dates = pl.date_range(start=pl.datetime(2020, 1, 1), end=pl.datetime(2020, 1, 5), interval='1d', eager=True)
        included_keys = list(range(1, 5))
        for key in included_keys:
            for dt in included_dates:
                conn.execute('insert into test values (?, ?, ?, ?)', [key, dt, dt.day, str(dt) + ' string'])

        df_all = duckdb_to_polars(conn.table('test'))

        for key in range(0, 6):
            for start in [datetime.datetime(2000, 1, 1), *included_dates, datetime.datetime(2030, 1, 1)]:
                for end in [datetime.datetime(2000, 1, 1), *included_dates, datetime.datetime(2030, 1, 1)]:
                    for include_start in [True, False]:
                        for include_end in [True, False]:
                            for sample_maxsize in [None, 2, 10]:
                                window = TimeWindow(start, end, include_start, include_end, sample_maxsize)
                                closed: Literal['left', 'right', 'both', 'none'] = \
                                            'both' if window.include_start and window.include_end else \
                                            'left' if window.include_start else \
                                            'right' if window.include_end else \
                                            'none'
                                result = strategy.get(conn, key, window, ['val1', 'val2'])
                                if key not in included_keys:
                                    assert result is None, f'Expected result None for key {key}'
                                else:
                                    assert result is not None
                                
                                    expected = df_all.filter(pl.col('key') == key).drop('key').filter(pl.col('date').is_between(window.start, window.end, closed=closed))
                                    if window.sample_maxsize is None:
                                        assert result is not None
                                        assert result.dataset.data.equals(expected)
                                    else:
                                        # Can't replicate sampling logic exactly, so can't predict rows that would be sampled
                                        window_without_sampling = TimeWindow(start, end, include_start, include_end, None)
                                        result_without_sampling = strategy.get(conn, key, window_without_sampling, ['val1', 'val2'])
                                        assert result_without_sampling is not None
                                        assert result.dataset.data.height == min(result_without_sampling.dataset.data.height, window.sample_maxsize)
                                        for row in result.dataset.data.iter_rows():
                                            assert row in result_without_sampling.dataset.data.rows()


        
if __name__ == '__main__':
    test_timeseries()
