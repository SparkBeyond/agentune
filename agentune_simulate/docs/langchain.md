# Configuring langchain models

This library uses langchain to define LLMs. These appear as instances of the class `BaseChatModel`;
langchain comes with many implementations of this class, e.g. `ChatOpenAI` for OpenAI-compatible models,
or `ChatAnthropic`.

If you are unfamiliar with langchain, you should be aware of some useful parameters you can pass to model instances:

## Retries and timeouts

This is a parameter specific to the OpenAI LLM provider; other providers such as Anthropic likely
have other, similar parameter, which may have different names and types.

The `timeout` parameter can be of type `httpx.Timeout`, which lets you set separate timeouts
for connecting, reading, writing, and getting an available connection from the connection pool.
It can also be of type `float`, which sets that value for all timeout types. In either case,
the underlying `float` timeout values are in minutes.

The `max_retries` parameter specifies how many times to retry an HTTP request that failed for transient reasons
(e.g. a connection error or a rate limit). There is some backoff between the retries.

The default value for both parameters is `None`, in which case the OpenAI client library's defaults
are used. [These are set here](https://github.com/openai/openai-python/blob/main/src/openai/_constants.py);
as of this writing, they are 2 retries (which is reasonable) and a 10 minute read timeout (which is usually
too high).

```python
llm = ChatOpenAI(..., timeout=30.0, max_retries=3)
```

## Caching

See the [Langchain cache documentation](https://python.langchain.com/api_reference/core/caches.html).
This is supported by all `BaseChatModel` implementations.

You can pass a `cache` parameter to a model instance to reuse previously seen replies to the same prompt.
An in-memory LRU cache, which lives only as long as the current Python process, can be configured with:

```python
from langchain_core.caches import InMemoryCache
llm=ChatOpenAI(..., cache=InMemoryCache(maxsize=1000))
```

There are many implementations of on-disk or in-database caching available in the `langchain-community` package;
you can see the [full list](https://python.langchain.com/api_reference/community/cache.html).
One of the simplest options is to use sqlite, which stores the cache in a single local file:

```python
from langchain_community.cache import SQLiteCache
llm=ChatOpenAI(..., cache=SQLiteCache(database_path='.cache.db'))
```

Note that sqlite has issues when the same file is accessed from multiple Python processes at once.
If you need to support that, consider using a different cache implementation from that list,
backed by a separate data storage process such as Redis or Postgresql.

## Rate limits

See the [Langchain rate limits documentation](https://python.langchain.com/docs/how_to/chat_model_rate_limiting/).
This is supported by all `BaseChatModel` implementations.

You can limit the rate at which the model will access its (presumably remote) implementation. This helps
you not run into your account's rate limit, and to catch accidental attempts to send many more requests
than intended.

Rate limiting applies to requests (e.g. HTTP requests) made behind the scenes. If the LLM provider's API
supports a batch endpoint, a single (batch) request counts as one request to the rate limiter.
Token rate limiting is not supported.

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=20, # overall rate
    max_bucket_size=15 # Max concurrent requests at any one point in time
)
llm=ChatOpenAI(..., rate_limiter=rate_limiter)
```
