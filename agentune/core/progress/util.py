from collections.abc import Awaitable

from agentune.core.progress.base import ProgressStage


async def execute_and_count[T](op: Awaitable[T], stage: ProgressStage | None) -> T:
    """Execute an operation and increment the stage count by 1 when it completes."""
    result = await op
    if stage is not None:
        stage.increment_count(1)
    return result
