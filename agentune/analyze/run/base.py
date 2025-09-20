from __future__ import annotations

import httpx
from attrs import frozen

from agentune.analyze.core.database import DuckdbManager
from agentune.analyze.core.llm import LLMContext
from agentune.analyze.core.sercontext import SerializationContext
from agentune.analyze.util.httpx_limit import AsyncLimitedTransport

default_max_concurrent_requests = 200 # Also affects the max open connections
default_max_keepalive_connections = 100
default_timeout = httpx.Timeout(timeout=600, connect=5.0)


def default_httpx_async_client(max_concurrent: int | None = default_max_concurrent_requests) -> httpx.AsyncClient:
    """Create a client configured with the same timeouts, connection limits, and redirects policy as the openai client library.

    Also enable http2; this will hopefully lead to much fewer connections in practice and better performance,
    but we can't rely on running in an environment where http2 isn't blocked by middleware for some reason,
    and not all (non-major) providers support http2.

    Args:
        max_concurrent:
            maximum number of concurrent http requests to allow.
    """
    # See openai._client._DefaultAsyncHttpxClient
    base = httpx.AsyncClient(http2=True, timeout=default_timeout,
                             follow_redirects=True, limits=httpx.Limits(max_connections=max_concurrent or default_max_concurrent_requests,
                                                                        max_keepalive_connections=max(default_max_keepalive_connections,
                                                                                                      max_concurrent or default_max_concurrent_requests)))
    # Don't pass the transport directly because httpx.AsyncClient can create different transport instances
    # based on other parameters
    if max_concurrent is not None:
        AsyncLimitedTransport.add_limits(base, max_concurrent)
    return base


@frozen
class RunContext:
    ser_context: SerializationContext
    ddb_manager: DuckdbManager
    
    @property
    def llm_context(self) -> LLMContext: return self.ser_context.llm_context

    @staticmethod
    def create_default_context(ddb_manager: DuckdbManager,
                               httpx_async_client: httpx.AsyncClient | None = None,
                               httpx_max_concurrent_requests: int | None = default_max_concurrent_requests) -> RunContext:
        """If httpx_async_client and httpx_max_concurrent_requests are both provided, the limit is applied to the given client.
        NOTE that this modifies that client instance.
        """
        if not httpx_async_client:
            httpx_async_client = default_httpx_async_client(httpx_max_concurrent_requests)
        elif httpx_max_concurrent_requests is not None:
            AsyncLimitedTransport.add_limits(httpx_async_client, httpx_max_concurrent_requests)
        llm_context = LLMContext(httpx_async_client)
        ser_context = SerializationContext(llm_context)
        return RunContext(ser_context, ddb_manager)
    
    async def aclose(self) -> None:
        self.ddb_manager.close()
        await self.llm_context.httpx_async_client.aclose()
