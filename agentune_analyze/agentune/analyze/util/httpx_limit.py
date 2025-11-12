from __future__ import annotations

from asyncio import BoundedSemaphore
from types import TracebackType
from typing import Self, cast, override

import httpx
from attrs import frozen
from httpx import AsyncBaseTransport

from agentune.analyze.util import asynclimits


@frozen(eq=False, hash=False)
class AsyncLimitedTransport(httpx.AsyncBaseTransport):
    """Limit the number of concurrent outstanding HTTP requests.

    Works for both http/1 and http/2. Timeouts and retries apply only once a request has been started, not while it
    is waiting.

    In the future we can add rate limits, and add limits that apply per request group and not globally.
    (Like httpx-ratelimiter (https://github.com/Midnighter/httpx-limiter) and some other libraries, but each of them
    has a different set of bugs and features, e.g. httpx-limiter doesn't override this class's methods other than
    handle_async_request; so we'll probably use pyrate directly.)
    """
    base_transport: AsyncBaseTransport
    semaphore: BoundedSemaphore

    @property
    def max_concurrent(self) -> int:
        return asynclimits.bounded_semaphore_limit(self.semaphore)

    @staticmethod
    def create(base_transport: AsyncBaseTransport, max_concurrent: int) -> AsyncLimitedTransport:
        return AsyncLimitedTransport(base_transport, BoundedSemaphore(max_concurrent))

    @staticmethod
    def add_limits(client: httpx.AsyncClient, max_concurrent: int) -> None:
        """Wrap the client's existing transport (and mounts, if any) to apply these limits.

        This modifies the client instance in-place.
        """
        semaphore = BoundedSemaphore(max_concurrent) # Reuse instance in case of mounts
        def wrap(transport: AsyncBaseTransport) -> AsyncBaseTransport:
            return AsyncLimitedTransport(transport, semaphore)
        client._transport = wrap(client._transport)
        client._mounts = { key: wrap(transport) if transport is not None else None for key, transport in client._mounts.items() }

    @override
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async with self.semaphore:
            return await self.base_transport.handle_async_request(request)

    @override
    async def aclose(self) -> None:
        return await self.base_transport.aclose()

    @override
    async def __aenter__(self) -> Self:
        return cast(Self, await self.base_transport.__aenter__())

    @override
    async def __aexit__(self, exc_type: type[BaseException] | None = None, exc_value: BaseException | None = None,
                        traceback: TracebackType | None = None) -> None:
        return await self.base_transport.__aexit__(exc_type, exc_value, traceback)


