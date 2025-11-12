"""Default values and component choices.

The top-level values defined in this module are used as defaults for function parameters,
so changing at runtime them has no effect.
"""
from __future__ import annotations

import httpx
from attrs import frozen

from agentune.analyze.api.base import RunContext
from agentune.analyze.core import default_duckdb_batch_size
from agentune.analyze.feature.compute.base import FeatureComputer
from agentune.analyze.feature.compute.universal import (
    UniversalAsyncFeatureComputer,
    UniversalSyncFeatureComputer,
)
from agentune.analyze.feature.gen.base import FeatureGenerator
from agentune.analyze.feature.gen.insightful_text_generator.insightful_text_generator import (
    ConversationQueryFeatureGenerator,
)
from agentune.analyze.feature.recommend import ConversationActionRecommender
from agentune.analyze.feature.select import FeatureSelector
from agentune.analyze.feature.select.base import EnrichedFeatureSelector
from agentune.analyze.feature.select.linear_pairwise import LinearPairWiseFeatureSelector
from agentune.analyze.run.analysis.base import (
    AnalyzeComponents,
    AnalyzeParams,
    AnalyzeRunner,
)
from agentune.analyze.run.analysis.impl import AnalyzeRunnerImpl
from agentune.analyze.run.enrich.base import EnrichRunner
from agentune.analyze.run.enrich.impl import EnrichRunnerImpl
from agentune.analyze.util.httpx_limit import AsyncLimitedTransport

# Defaults similar to those of the openai client library
default_max_concurrent_requests = 200 # Also affects the max open connections for HTTP/1
default_max_keepalive_connections = 100
default_timeout = httpx.Timeout(timeout=600, connect=5.0)


def create_default_httpx_async_client(max_concurrent: int = default_max_concurrent_requests,
                                      max_keepalive: int = default_max_keepalive_connections,
                                      timeout: httpx.Timeout = default_timeout,) -> httpx.AsyncClient:
    """Create a new HTTPX client. This creates a connection pool which keeps some connections open even when idle.

    Remember to close the client instance.

    Besides the parameters given, we always enable http2; this will hopefully lead to much fewer connections in practice
    and better performance, but we can't rely on running in an environment where http2 isn't blocked by middleware for
    some reason, and not all (non-major) providers support http2.

    Args:
        max_concurrent:
            maximum number of concurrent http requests to allow. This is also the maximum number of concurrent HTTP/1 connections,
            but in HTTP/2 mode separate connections are not needed for concurrent requests.
        max_keepalive: maximum number of open connections to keep alive while idle.
        timeout: timeouts for connecting to servers and getting responses; can be overriden by library code on a per-call basis.
    """
    # See openai._client._DefaultAsyncHttpxClient
    base = httpx.AsyncClient(http2=True, timeout=timeout,
                             follow_redirects=True, limits=httpx.Limits(max_connections=max_concurrent,
                                                                        max_keepalive_connections=max(max_keepalive, max_concurrent)))
    # Don't pass the transport directly because httpx.AsyncClient can create different transport instances
    # based on other parameters
    if max_concurrent is not None:
        AsyncLimitedTransport.add_limits(base, max_concurrent)
    return base

@frozen
class BoundDefaults:
    """Default values for various component classes and parameters."""
    run_context: RunContext

    @property
    def duckdb_batch_size(self) -> int:
        return default_duckdb_batch_size

    def analyze_params(self) -> AnalyzeParams:
        return AnalyzeParams()

    def conversation_query_feature_generator(self) -> ConversationQueryFeatureGenerator:
        return ConversationQueryFeatureGenerator.default(self.run_context._llm_context)

    def feature_generators(self) -> tuple[FeatureGenerator, ...]:
        return (self.conversation_query_feature_generator(), )

    def feature_selector(self) -> FeatureSelector | EnrichedFeatureSelector:
        return LinearPairWiseFeatureSelector()

    def feature_computers(self) -> tuple[type[FeatureComputer], ...]:
        return (UniversalAsyncFeatureComputer, UniversalSyncFeatureComputer)

    def enrich_runner(self) -> EnrichRunner:
        return EnrichRunnerImpl()

    def analyze_components(self) -> AnalyzeComponents:
        return AnalyzeComponents(self.feature_generators(), self.feature_selector(), self.feature_computers())

    def analyze_runner(self) -> AnalyzeRunner:
        return AnalyzeRunnerImpl()

    def conversation_action_recommender(self) -> ConversationActionRecommender:
        return ConversationActionRecommender.default(self.run_context._llm_context)
