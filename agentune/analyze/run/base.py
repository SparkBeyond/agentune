from __future__ import annotations

import httpx
from attrs import frozen

from agentune.analyze.core.database import DuckdbManager
from agentune.analyze.core.llm import LLMContext
from agentune.analyze.core.sercontext import SerializationContext


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
            httpx_async_client = httpx.AsyncClient(http2=True)
        llm_context = LLMContext(httpx_async_client)
        ser_context = SerializationContext(llm_context)
        return RunContext(ser_context, ddb_manager)
    
    async def aclose(self) -> None:
        self.ddb_manager.close()
        await self.llm_context.httpx_async_client.aclose()
