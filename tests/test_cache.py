"""
Regression tests for the caching layer.

Validates TTLCache get/set/expire and the cached decorators.
"""

import time

import pytest

from backend.core.cache import TTLCache, embedding_cache, llm_cache

pytestmark = pytest.mark.unit


class TestTTLCache:
    def test_set_and_get(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_miss_returns_none(self):
        cache = TTLCache(ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_expiration(self):
        cache = TTLCache(ttl_seconds=0)  # expires immediately
        cache.set("key1", "value1")
        time.sleep(0.01)
        assert cache.get("key1") is None

    def test_clear(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_max_size_eviction(self):
        cache = TTLCache(ttl_seconds=60, max_size=10)
        for i in range(15):
            cache.set(f"key_{i}", f"value_{i}")
        # After eviction, cache should have at most max_size entries
        assert len(cache._cache) <= 10

    def test_make_key_deterministic(self):
        cache = TTLCache(ttl_seconds=60)
        key1 = cache._make_key("a", "b", x=1, y=2)
        key2 = cache._make_key("a", "b", y=2, x=1)  # different order
        assert key1 == key2  # kwargs order shouldn't matter


class TestCachedLLMCall:
    def test_caches_result(self):
        call_count = 0

        from backend.core.cache import cached_llm_call

        @cached_llm_call
        def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"response_to_{prompt}"

        # Clear cache before test
        llm_cache.clear()

        result1 = mock_llm("hello")
        result2 = mock_llm("hello")

        assert result1 == result2
        assert call_count == 1  # Only called once due to cache

    def test_different_args_different_cache(self):
        call_count = 0

        from backend.core.cache import cached_llm_call

        @cached_llm_call
        def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"response_to_{prompt}"

        llm_cache.clear()

        mock_llm("hello")
        mock_llm("world")

        assert call_count == 2  # Different args = different cache entries


class TestCachedEmbedding:
    def test_caches_embedding(self):
        call_count = 0

        from backend.core.cache import cached_embedding

        @cached_embedding
        def mock_embed(text: str) -> list[float]:
            nonlocal call_count
            call_count += 1
            return [0.1, 0.2, 0.3]

        embedding_cache.clear()

        result1 = mock_embed("test text")
        result2 = mock_embed("test text")

        assert result1 == result2
        assert call_count == 1
