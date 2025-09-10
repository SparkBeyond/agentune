from __future__ import annotations

import httpx
from attrs import frozen

from agentune.analyze.core.database import DuckdbManager
from agentune.analyze.core.llm import LLMContext
from agentune.analyze.core.sercontext import SerializationContext


def default_httpx_async_client() -> httpx.AsyncClient:
    """Create a client configured with the same timeouts, connection limits, and redirects policy as the openai client library.

    Also enable http2; this will hopefully lead to much fewer connections in practice and better performance,
    but we can't rely on running in an environment where http2 isn't blocked by middleware for some reason.
    """
    # See openai._client._DefaultAsyncHttpxClient
    return httpx.AsyncClient(http2=True, timeout=httpx.Timeout(timeout=600, connect=5.0),
                             follow_redirects=True, limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100))


@frozen
class RunContext:
    ser_context: SerializationContext
    ddb_manager: DuckdbManager
    
    @property
    def llm_context(self) -> LLMContext: return self.ser_context.llm_context

    @staticmethod
    def create_default_context(ddb_manager: DuckdbManager,
                               httpx_async_client: httpx.AsyncClient | None = None) -> RunContext:
        if not httpx_async_client:
            httpx_async_client = default_httpx_async_client()
        llm_context = LLMContext(httpx_async_client)
        ser_context = SerializationContext(llm_context)
        return RunContext(ser_context, ddb_manager)
    
    async def aclose(self) -> None:
        self.ddb_manager.close()
        await self.llm_context.httpx_async_client.aclose()
