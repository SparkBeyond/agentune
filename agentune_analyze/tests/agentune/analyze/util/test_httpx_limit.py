import asyncio

import httpx
from httpx import AsyncBaseTransport

from agentune.analyze.util.atomic import AtomicInt
from agentune.analyze.util.httpx_limit import AsyncLimitedTransport


def delay_transport(delay: float, max_concurrent: AtomicInt) -> AsyncBaseTransport:
    concurrent = AtomicInt()
    async def handler(_request: httpx.Request) -> httpx.Response:
        concurrent.inc_and_get()
        await asyncio.sleep(delay)
        max_concurrent.setmax(concurrent.get())
        concurrent.inc_and_get(-1)
        return httpx.Response(200, json={'text': 'Hello, world!'})
    return httpx.MockTransport(handler)


async def test_request_limit_enforcement() -> None:
    max_concurrent = AtomicInt()
    mock_transport = delay_transport(0.01, max_concurrent)
    limited_transport = AsyncLimitedTransport.create(mock_transport, max_concurrent=3)
    
    async with httpx.AsyncClient(transport=limited_transport) as client:
        # Start 10 requests concurrently
        tasks = [client.get('http://example.com') for _ in range(10)]
        responses = await asyncio.gather(*tasks)
    
    assert len(responses) == 10
    assert all(response.status_code == 200 for response in responses)
    
    assert max_concurrent.get() == 3
