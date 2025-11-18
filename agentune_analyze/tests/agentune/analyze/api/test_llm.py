import datetime
from collections.abc import Awaitable
from datetime import timedelta
from pathlib import Path

import pytest

from agentune.analyze.api.base import LlmCacheOnDisk, RunContext
from agentune.analyze.core.llm import LLMSpec


async def timed[T](op: Awaitable[T]) -> tuple[T, timedelta]:
    start = datetime.datetime.now()
    result = await op
    end = datetime.datetime.now()
    return result, end - start

@pytest.mark.integration
async def test_llm_cache(tmp_path: Path) -> None:
    cachefile = tmp_path / 'cache.sqlite'
    spec = LLMSpec('openai', 'gpt-4.1-nano')
    prompt = 'The meaning of life is '

    assert not cachefile.exists(), "Cache doesn't exist to begin with"
    async with await RunContext.create(llm_cache=LlmCacheOnDisk(cachefile, 100000)) as ctx:
        assert cachefile.exists(), 'Cache created'

        assert ctx.llm.cache_backend is not None
        assert len(ctx.llm.cache_backend) == 0

        llm = ctx.llm.get(spec)
        response1, time1 = await timed(llm.acomplete(prompt))
        assert time1 > datetime.timedelta(seconds=0.3)
        assert len(ctx.llm.cache_backend) == 1

        response2, time2 = await timed(llm.acomplete(prompt))
        assert response2 == response1, 'Cached response used'
        assert len(ctx.llm.cache_backend) == 1, 'Cached response used'
        assert time2 < datetime.timedelta(seconds=0.01)

    assert cachefile.exists(), 'Cache still exists after context is closed'

    async with await RunContext.create(llm_cache=LlmCacheOnDisk(cachefile, 100000)) as ctx:
        assert ctx.llm.cache_backend is not None
        assert len(ctx.llm.cache_backend) == 1

        llm = ctx.llm.get(spec)
        response3, time3 = await timed(llm.acomplete(prompt))
        assert response3 == response1, 'Cached response used'
        assert time3 < datetime.timedelta(seconds=0.01)
        assert len(ctx.llm.cache_backend) == 1, 'Cached response used again'

        ctx.llm.clear_cache()
        assert len(ctx.llm.cache_backend) == 0, 'Cached cleared'

        response4, time4 = await timed(llm.acomplete(prompt))
        assert len(ctx.llm.cache_backend) == 1, 'Cached new response'
        assert response4 != response1, 'Cached response used'
        assert time4 > datetime.timedelta(seconds=0.3)

        response5, time5 = await timed(llm.acomplete(prompt))
        assert response5 == response4, 'New cached response used'
        assert len(ctx.llm.cache_backend) == 1
        assert time5 < datetime.timedelta(seconds=0.01)

