from collections.abc import AsyncIterator

import pytest

from agentune.api.base import RunContext


@pytest.fixture
async def ctx() -> AsyncIterator[RunContext]:
    async with await RunContext.create() as ctx:
        yield ctx

