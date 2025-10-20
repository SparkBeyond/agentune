import asyncio
import logging
import time
from typing import cast

import pytest
from more_itertools.more import is_sorted

from agentune.analyze.progress.base import (
    ProgressStage,
    current_stage,
    root_stage_scope,
    stage_scope,
)

_logger = logging.getLogger(__name__)

def test_basics() -> None:
    assert current_stage() is None

    with stage_scope('foo') as foo:
        assert current_stage() is foo

        assert foo.name == 'foo'
        assert foo.count is None
        assert foo.total is None
        assert foo.parent is None
        assert foo.root is foo
        assert foo.children == ()
        assert not foo.is_completed

        foo.set_count(10)
        assert foo.count == 10
        foo.set_total(20)
        assert foo.total == 20

        with pytest.raises(ValueError, match='> total'):
            foo.set_count(30)
        with pytest.raises(ValueError, match='< count'):
            foo.set_total(5)

        with stage_scope('bar', 1, 10) as bar:
            assert current_stage() is bar
            assert bar.parent is foo
            assert bar.root is foo
            assert foo.children == (bar,)

            assert bar.name == 'bar'
            assert bar.count == 1
            assert bar.total == 10
            assert not bar.is_completed
            assert bar.started > foo.started
            assert bar.children == ()

        assert bar.is_completed, 'Stage completed when exiting scope'
        assert bar.total == bar.count == 1, 'Total set to count when completing scope'
        assert not foo.is_completed
        assert bar.completed is not None
        assert bar.completed > bar.started

        with root_stage_scope('new_root') as new_root:
            assert current_stage() is new_root
            assert new_root.parent is None
            assert new_root.children == ()
            assert new_root not in foo.children
            assert new_root.root is new_root
            assert not foo.is_completed

            foo.complete()
            assert not new_root.is_completed

        assert current_stage().is_completed
        with pytest.raises(ValueError, match='completed'):
            with stage_scope('new'): pass

    with stage_scope('one') as one:
        with stage_scope('two') as two:
            one.complete()
            assert one.is_completed
            assert two.is_completed
            two.complete() # Completing a stage twice does nothing

            with pytest.raises(ValueError, match='completed'):
                two.set_count(10)
            with pytest.raises(ValueError, match='completed'):
                two.set_total(10)

async def test_progress_reporter() -> None:
    async def apublish() -> None:
        with stage_scope('async', total=49) as stage:
            for i in range(50):
                stage.set_count(i)
                await asyncio.sleep(0.01)

    def publish() -> None:
        with stage_scope('sync', total=49) as stage:
            for i in range(50):
                stage.set_count(i)
                time.sleep(0.01)

    async def reporter() -> list[ProgressStage]:
        stages = []
        try:
            while True:
                stages.append(cast(ProgressStage, current_stage()).root.deepcopy())
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            return stages

    with stage_scope('root'):
        reporter_task = asyncio.create_task(reporter())
        await asyncio.sleep(0)

        await asyncio.gather(apublish(), asyncio.to_thread(publish))

        reporter_task.cancel()
        stages = await reporter_task

    assert len(stages) > 3 and len(stages) < 7, 'Sanity check'
    for stage in stages:
        assert stage.name == 'root'
    assert stages[0].children == ()
    for stage in stages[1:]:
        assert len(stage.children) == 2
        for child in stage.children:
            assert child.name in {'async', 'sync'}
            assert child.total == 49

    for name in ('async', 'sync'):
        counts = [child.count for stage in stages[1:] for child in stage.children if child.name == name]
        assert is_sorted(counts)

