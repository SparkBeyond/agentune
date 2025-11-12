from __future__ import annotations

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, CompletionResponse

from agentune.analyze.llmcache.base import LLMCacheKey
from agentune.analyze.llmcache.serializing_kvstore import SerializingKVStore


def test_serializing_mapping() -> None:
    """Test SerializingMapping serialization, storage, and retrieval."""
    inner: dict[bytes, bytes] = {}
    cache = SerializingKVStore(inner=inner)
    
    # Test responses and keys
    chat_resp = ChatResponse(message=ChatMessage(role='assistant', content='Hello'), raw={})
    comp_resp = CompletionResponse(text='World', raw={})
    key1 = LLMCacheKey(messages=(ChatMessage(role='user', content='Hi'),), prompt=None, formatted=False, kwargs={})
    key2 = LLMCacheKey(messages=(), prompt='Different', formatted=True, kwargs={})
    
    # Basic operations: store both types, verify length, retrieve correct types
    cache[key1] = chat_resp
    cache[key2] = comp_resp
    assert len(cache) == 2
    retrieved1 = cache[key1]
    retrieved2 = cache[key2]
    assert isinstance(retrieved1, ChatResponse) and retrieved1.message.content == 'Hello'
    assert isinstance(retrieved2, CompletionResponse) and retrieved2.text == 'World'
    
    # Verify serialization actually happened (stored as JSON strings)
    assert len(inner) == 2 and all(isinstance(v, bytes) for v in inner.values())
    
    # Key hashing: identical keys should work interchangeably
    key1_dup = LLMCacheKey(messages=(ChatMessage(role='user', content='Hi'),), prompt=None, formatted=False, kwargs={})
    retrieved_dup = cache[key1_dup]
    assert isinstance(retrieved_dup, ChatResponse) and retrieved_dup.message.content == 'Hello'
    
    # Error conditions: missing keys, iteration not implemented
    missing_key = LLMCacheKey(messages=(), prompt='Missing', formatted=False, kwargs={})
    with pytest.raises(KeyError):
        _ = cache[missing_key]
    with pytest.raises(KeyError):
        del cache[missing_key]

    # Deletion and updates
    del cache[key1]
    assert len(cache) == 1
    cache[key2] = chat_resp  # Replace CompletionResponse with ChatResponse
    assert isinstance(cache[key2], ChatResponse)


def test_complex_serialization() -> None:
    """Test serialization with complex data structures."""
    cache = SerializingKVStore(inner={})
    
    # Complex response with nested structures and special characters
    response = ChatResponse(
        message=ChatMessage(role='assistant', content='Response with "quotes" and \\backslashes\\'),
        raw={'model': 'test', 'usage': {'tokens': 42}, 'metadata': [1, 2, 3]}
    )
    key = LLMCacheKey(
        messages=(
            ChatMessage(role='system', content='System prompt'),
            ChatMessage(role='user', content='User query with special chars: "test"')
        ),
        prompt='completion prompt',
        formatted=True,
        kwargs={'temperature': 0.7, 'max_tokens': 100}
    )
    
    cache[key] = response
    retrieved = cache[key]
    
    # Verify data integrity after serialization round-trip
    assert isinstance(retrieved, ChatResponse)
    assert retrieved.message.content == response.message.content
    assert retrieved.raw == response.raw
