"""
Caching layer for LLM responses and embeddings.

Uses in-memory cache with TTL for free, simple caching.
For production, consider Redis or Memcached.
"""

import hashlib
import json
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class TTLCache:
    """Simple in-memory cache with time-to-live."""

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: dict[str, tuple[Any, float]] = {}

    def _make_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate cache key from function arguments."""
        key_data = {
            "args": args,
            "kwargs": kwargs,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]

        # Check if expired
        if time.time() - timestamp > self.ttl_seconds:
            del self._cache[key]
            return None

        logger.debug("cache_hit", key=key[:16])
        return value

    def set(self, key: str, value: Any) -> None:
        """Set value in cache with current timestamp."""
        # Evict oldest entries if cache is full
        if len(self._cache) >= self.max_size:
            # Remove oldest 10% of entries
            sorted_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k][1])
            for old_key in sorted_keys[: self.max_size // 10]:
                del self._cache[old_key]

        self._cache[key] = (value, time.time())
        logger.debug("cache_set", key=key[:16])

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        logger.info("cache_cleared")


# Global cache instances
llm_cache = TTLCache(ttl_seconds=3600, max_size=500)  # 1 hour TTL for LLM responses
embedding_cache = TTLCache(ttl_seconds=86400, max_size=1000)  # 24 hour TTL for embeddings


def cached_llm_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to cache LLM responses.

    Caches based on function name and arguments.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Generate cache key
        cache_key = f"{func.__name__}:{llm_cache._make_key(*args, **kwargs)}"

        # Try to get from cache
        cached_result = llm_cache.get(cache_key)
        if cached_result is not None:
            logger.info("llm_cache_hit", function=func.__name__)
            return cached_result

        # Call function
        result = func(*args, **kwargs)

        # Store in cache
        llm_cache.set(cache_key, result)
        logger.info("llm_cache_miss", function=func.__name__)

        return result

    return wrapper


def cached_embedding(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to cache embedding computations.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Generate cache key
        cache_key = f"{func.__name__}:{embedding_cache._make_key(*args, **kwargs)}"

        # Try to get from cache
        cached_result = embedding_cache.get(cache_key)
        if cached_result is not None:
            logger.info("embedding_cache_hit", function=func.__name__)
            return cached_result

        # Call function
        result = func(*args, **kwargs)

        # Store in cache
        embedding_cache.set(cache_key, result)
        logger.info("embedding_cache_miss", function=func.__name__)

        return result

    return wrapper
