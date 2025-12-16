"""Tests for custom ProgressReporter implementations and lifecycle."""
import asyncio
from datetime import timedelta
from typing import override

import pytest

from agentune.api.base import RunContext
from agentune.core.progress.base import ProgressStage
from agentune.core.progress.reporters.base import ProgressReporter, progress_setup


class RecordingReporter(ProgressReporter):
    """A test reporter that records all calls."""
    
    def __init__(self, poll_interval: timedelta = timedelta(seconds=0.05)) -> None:
        self.poll_interval = poll_interval
        self.start_calls: list[ProgressStage] = []
        self.update_calls: list[ProgressStage] = []
        self.stop_called = False
    
    @override
    async def start(self, root_stage: ProgressStage) -> None:
        self.start_calls.append(root_stage.deepcopy())
    
    @override
    async def update(self, snapshot: ProgressStage) -> None:
        self.update_calls.append(snapshot.deepcopy())
    
    @override
    async def stop(self) -> None:
        self.stop_called = True


@pytest.fixture
def recording_reporter() -> RecordingReporter:
    """Provide a fresh RecordingReporter."""
    return RecordingReporter()


async def test_reporter_receives_start_and_updates(recording_reporter: RecordingReporter) -> None:
    """Reporter receives start() and update() calls."""
    async with progress_setup(recording_reporter):
        await asyncio.sleep(0.1)
    
    assert len(recording_reporter.start_calls) == 1
    assert recording_reporter.start_calls[0].name == 'RunContext Root'
    assert len(recording_reporter.update_calls) >= 1
    assert recording_reporter.stop_called

async def test_custom_reporter_in_runcontext(recording_reporter: RecordingReporter) -> None:
    """Custom ProgressReporter instance can be passed to RunContext.create."""
    async with await RunContext.create(progress_reporter=recording_reporter):
        await asyncio.sleep(0.1)
    
    assert len(recording_reporter.start_calls) == 1
    assert not recording_reporter.stop_called, 'User-provided reporter should not be stopped'


async def test_disabled_progress_reporter() -> None:
    """progress_reporter=None disables progress reporting."""
    async with await RunContext.create(progress_reporter=None):
        pass

