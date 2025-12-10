from __future__ import annotations

import asyncio
import contextlib
import typing
from abc import ABC, abstractmethod
from datetime import timedelta

from agentune.core.progress.base import ProgressStage, root_stage_scope


class ProgressReporter(ABC):
    """Abstract base class for all progress reporters.
    
    Reporters receive progress updates and display them in various formats
    (console, logs, etc.). They are activated when root stages appear and
    receive periodic snapshot updates during operation execution.
    
    All lifecycle methods are async to ensure compatibility with async operations.
    """
    poll_interval: timedelta

    @abstractmethod
    async def start(self, root_stage: ProgressStage) -> None:
        """Called when a root stage is created and reporting should begin.
        
        Args:
            root_stage: The root progress stage that will be observed.
        """
        ...

    @abstractmethod
    async def update(self, snapshot: ProgressStage) -> None:
        """Called periodically with progress snapshot updates.
        
        Args:
            snapshot: A snapshot of the current progress stage tree.
                      This is a deep copy that can be safely accessed
                      without locking.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Called when the root stage completes and reporting should end.
        
        This method should perform any necessary cleanup, such as closing
        displays or flushing buffers.
        """
        ...

@contextlib.asynccontextmanager
async def progress_setup(
    progress_reporter: ProgressReporter | None,
    root_stage_name: str = 'RunContext Root',
    owns_reporter: bool = True,
) -> typing.AsyncGenerator[None, None]:
    """Async context manager for progress reporting setup and cleanup.
    
    Args:
        progress_reporter: The reporter to use, or None to disable progress reporting.
        root_stage_name: Name for the root progress stage.
        owns_reporter: If True, stop() will be called on the reporter when exiting.
                       Set to False when the caller owns the reporter lifecycle.
    """
    with root_stage_scope(root_stage_name) as root_stage_instance:
        if progress_reporter is not None:
            await progress_reporter.start(root_stage_instance.deepcopy())
        
        polling_task: asyncio.Task[None] | None = None
        if progress_reporter is not None:
            async def poll_progress() -> None:
                """Poll for progress updates and send them to the reporter."""
                while True:
                    snapshot = root_stage_instance.deepcopy()
                    await progress_reporter.update(snapshot)
                    if snapshot.is_completed:
                        break
                    await asyncio.sleep(progress_reporter.poll_interval.total_seconds())
            
            polling_task = asyncio.create_task(poll_progress())
        
        try:
            yield
        finally:
            # Stop polling task
            if polling_task is not None:
                polling_task.cancel()
            # Stop reporter only if we own it
            if progress_reporter is not None and owns_reporter:
                await progress_reporter.stop()

