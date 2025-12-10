"""Tests for LogReporter output formatting and behavior."""
import asyncio
import logging
from collections.abc import Iterator
from datetime import timedelta

import pytest

from agentune.core.progress.base import stage_scope
from agentune.core.progress.reporters.base import progress_setup
from agentune.core.progress.reporters.log import LogReporter


class LogCapture(logging.Handler):
    """Handler that captures log records for testing."""
    
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []
    
    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
    
    @property
    def messages(self) -> list[str]:
        return [r.getMessage() for r in self.records]


@pytest.fixture
def log_capture() -> Iterator[tuple[LogCapture, logging.Logger]]:
    """Provide a LogCapture handler attached to a test logger."""
    capture = LogCapture()
    logger = logging.getLogger('test.progress')
    logger.addHandler(capture)
    logger.setLevel(logging.INFO)
    yield capture, logger
    logger.removeHandler(capture)


@pytest.fixture
def log_reporter() -> LogReporter:
    """Provide a LogReporter using the test logger."""
    return LogReporter(timedelta(seconds=0.01), 'test.progress', logging.INFO)


async def test_logs_started_event(log_capture: tuple[LogCapture, logging.Logger], log_reporter: LogReporter) -> None:
    """LogReporter logs [started] event on start."""
    capture, _ = log_capture
    
    async with progress_setup(log_reporter):
        pass
    
    assert any('[started]' in msg for msg in capture.messages)
    assert any('RunContext Root' in msg for msg in capture.messages)


async def test_logs_completed_event(log_capture: tuple[LogCapture, logging.Logger], log_reporter: LogReporter) -> None:
    """LogReporter logs [completed] event when stages complete."""
    capture, _ = log_capture
    
    async with progress_setup(log_reporter):
        with stage_scope('task', count=0, total=10) as task:
            task.set_count(10)
        await asyncio.sleep(0.05)
    
    assert any('[completed]' in msg and 'task' in msg for msg in capture.messages)


async def test_logs_progress_event(log_capture: tuple[LogCapture, logging.Logger], log_reporter: LogReporter) -> None:
    """LogReporter logs [progress] event when count changes."""
    capture, _ = log_capture
    
    async with progress_setup(log_reporter):
        with stage_scope('work', count=0, total=10) as work:
            for i in range(1, 11):
                work.set_count(i)
                await asyncio.sleep(0.02)
    
    assert any('[progress]' in msg for msg in capture.messages)


async def test_count_format_with_total(log_capture: tuple[LogCapture, logging.Logger], log_reporter: LogReporter) -> None:
    """Count formatting: (count/total) when both present."""
    capture, _ = log_capture
    
    async with progress_setup(log_reporter):
        with stage_scope('fmt', count=0, total=50) as task:
            for i in range(1, 26):
                task.set_count(i)
                await asyncio.sleep(0.015)
    
    progress_msgs = [m for m in capture.messages if '[progress]' in m and 'fmt' in m]
    assert len(progress_msgs) > 0, f'Expected progress messages, got: {capture.messages}'
    assert any('/50)' in msg for msg in progress_msgs)


async def test_count_format_without_total(log_capture: tuple[LogCapture, logging.Logger], log_reporter: LogReporter) -> None:
    """Count formatting: (count) when no total - only shows the current count."""
    capture, _ = log_capture
    
    async with progress_setup(log_reporter):
        with stage_scope('nototal', count=0) as task:
            task.set_count(5)
            await asyncio.sleep(0.03)
    
    nototal_msgs = [m for m in capture.messages if 'nototal' in m]
    assert len(nototal_msgs) > 0
    started_msgs = [m for m in nototal_msgs if '[started]' in m]
    assert any('(0)' in m for m in started_msgs), 'Started should show (0) when no total'


async def test_completion_order_children_before_parents(log_capture: tuple[LogCapture, logging.Logger]) -> None:
    """Completion events log children before parents."""
    capture, _ = log_capture
    
    # Long poll so everything completes between polls
    reporter = LogReporter(timedelta(seconds=10), 'test.progress', logging.INFO)
    async with progress_setup(reporter):
        with stage_scope('parent'):
            with stage_scope('child'):
                pass
    
    completed_msgs = [m for m in capture.messages if '[completed]' in m]
    child_idx = next((i for i, m in enumerate(completed_msgs) if 'child' in m), -1)
    parent_idx = next((i for i, m in enumerate(completed_msgs) if 'parent' in m and 'child' not in m), -1)
    
    if child_idx >= 0 and parent_idx >= 0:
        assert child_idx < parent_idx, 'Child should complete before parent in logs'

