from __future__ import annotations

import pytest

from agentune.analyze.util.lrucache import LRUCache

# ruff: noqa: PT018


def test_basic_functionality() -> None:
    """Test basic cache operations."""
    cache = LRUCache[str, int](maxsize=2)
    
    # Basic set/get/contains/len
    cache['a'] = 1
    cache['b'] = 2
    assert cache['a'] == 1 and cache['b'] == 2
    assert 'a' in cache and 'c' not in cache
    assert len(cache) == 2
    
    # Get method with defaults
    assert cache.get('a') == 1 and cache.get('c') is None
    assert cache.get('c', 99) == 99


def test_lru_eviction() -> None:
    """Test LRU eviction and access order tracking."""
    cache = LRUCache[str, int](maxsize=2)
    cache['a'] = 1
    cache['b'] = 2
    
    # Access 'a' to make it most recent, then add 'c'
    _ = cache['a']
    cache['c'] = 3
    assert 'a' in cache and 'c' in cache and 'b' not in cache
    
    # Update existing key should move to end
    cache['a'] = 10
    cache['d'] = 4
    cache['e'] = 5
    assert 'a' not in cache and 'c' not in cache and 'd' in cache and 'e' in cache


def test_edge_cases() -> None:
    """Test edge cases and error conditions."""
    # Zero capacity
    cache = LRUCache[str, int](maxsize=0)
    cache['a'] = 1
    assert len(cache) == 0 and 'a' not in cache
    
    # Single capacity  
    cache = LRUCache[str, int](maxsize=1)
    cache['a'] = 1
    cache['b'] = 2
    assert len(cache) == 1 and 'b' in cache and 'a' not in cache
    
    # Error conditions
    with pytest.raises(KeyError):
        _ = cache['missing']
    with pytest.raises(KeyError):
        del cache['missing']


def test_mutation_operations() -> None:
    """Test deletion, clearing, and iteration."""
    cache = LRUCache[str, int](maxsize=3)
    cache.update({'a': 1, 'b': 2, 'c': 3})
    
    # Deletion
    del cache['b']
    assert len(cache) == 2 and 'b' not in cache
    
    # Iteration reflects order
    keys_before = list(cache)
    _ = cache['a']  # Move 'a' to end
    keys_after = list(cache)
    assert keys_before != keys_after and keys_after[-1] == 'a'
    
    # Clear
    cache.clear()
    assert len(cache) == 0 and list(cache) == []


def test_types_and_mapping() -> None:
    """Test different types and mapping protocol."""
    cache = LRUCache[int, str](maxsize=2)
    cache[1] = 'one'
    cache[2] = 'two'
    
    assert set(cache.keys()) == {1, 2}
    assert len(cache) == 2
    assert cache[1] == 'one'
