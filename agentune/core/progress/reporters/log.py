from __future__ import annotations

import datetime
import logging
from datetime import timedelta
from typing import override

from agentune.core.progress.base import ProgressStage, StageDiff, root_stage
from agentune.core.progress.reporters.base import ProgressReporter


class LogReporter(ProgressReporter):
    """Progress reporter that logs progress updates with Python's standard logging module.
    The logging output destination is controlled by the user's logging module configuration.
    """

    def __init__(
        self,
        poll_interval: timedelta,
        logger_name: str,
        log_level: int,
    ) -> None:
        self.poll_interval = poll_interval
        self._logger = logging.getLogger(logger_name)
        self._log_level = log_level
        self._previous_snapshot: ProgressStage | None = None

    @override
    async def start(self, root_stage: ProgressStage) -> None:
        """Start logging progress for the given root stage."""
        snapshot = root_stage.deepcopy()
        self._log_stage_tree(snapshot, (snapshot.name,))
        self._previous_snapshot = snapshot

    def _log_stage_tree(self, stage: ProgressStage, path: tuple[str, ...]) -> None:
        """Recursively log a stage and all its children as started."""
        start_count = 0 if stage.count is not None else None
        self._log_stage_event(path, 'started', start_count, stage.total)
        for child in stage.children:
            self._log_stage_tree(child, (*path, child.name))

    @override
    async def update(self, snapshot: ProgressStage) -> None:
        """Log progress updates."""
        if self._previous_snapshot is None:
            return
        
        diffs = ProgressStage.diff(self._previous_snapshot, snapshot)
        
        # Collect all events with timestamps and sort chronologically
        events: list[tuple[datetime.datetime, tuple[str, ...], str, int | None, int | None]] = []
        now = datetime.datetime.now()
        for diff in diffs:
            if diff.added:
                start_count = 0 if diff.new_count is not None else None
                events.append((diff.started_at or now, diff.path, 'started', start_count, diff.new_total))
            if diff.completed:
                events.append((diff.completed_at or now, diff.path, 'completed', diff.new_count, diff.new_total))
            elif not diff.added and (diff.count_changed or diff.total_changed):
                # Skip progress events where count == total (will be logged as completed later)
                if diff.new_count != diff.new_total:
                    events.append((now, diff.path, 'progress', diff.new_count, diff.new_total))
        
        for _, path, event, count, total in sorted(events, key=lambda e: e[0]):
            self._log_stage_event(path, event, count, total)
        
        self._previous_snapshot = snapshot.deepcopy()

    @override
    async def stop(self) -> None:
        """Stop logging and perform cleanup."""
        current_root = root_stage()
        if current_root is not None:
            final_snapshot = current_root.deepcopy()
            await self.update(final_snapshot)
        
        self._previous_snapshot = None

    def _log_stage_event(
        self,
        path: tuple[str, ...],
        event: str,
        count: int | None,
        total: int | None,
    ) -> None:
        """Log a stage event with optional count/total."""
        path_str = ' > '.join(path)
        
        if count is not None and total is not None:
            msg = f'{path_str} ({count}/{total}) [{event}]'
        elif count is not None:
            msg = f'{path_str} ({count}) [{event}]'
        elif total is not None:
            msg = f'{path_str} (0/{total}) [{event}]'
        else:
            msg = f'{path_str} [{event}]'
        
        self._logger.log(self._log_level, msg)

