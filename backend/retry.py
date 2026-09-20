"""
Retry utilities with exponential backoff for resilient external API calls.

Uses tenacity for configurable retry policies.
"""

import logging
from collections.abc import Callable
from typing import TypeVar

import structlog
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

T = TypeVar("T")


def create_retry_decorator(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exponential_base: float = 2.0,
    retry_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        IOError,
    ),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Create a retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including first)
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exponential_base: Base for exponential backoff
        retry_exceptions: Exception types to retry on

    Returns:
        Decorator function
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=min_wait,
            max=max_wait,
            exp_base=exponential_base,
        ),
        retry=retry_if_exception_type(retry_exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO),
        reraise=True,
    )


# Pre-configured retry decorators for common use cases

# For LLM API calls - more retries, longer waits
llm_retry = create_retry_decorator(
    max_attempts=3,
    min_wait=2.0,
    max_wait=30.0,
    retry_exceptions=(
        ConnectionError,
        TimeoutError,
        IOError,
    ),
)

# For vector store operations - fewer retries, shorter waits
vector_store_retry = create_retry_decorator(
    max_attempts=2,
    min_wait=0.5,
    max_wait=5.0,
    retry_exceptions=(
        ConnectionError,
        TimeoutError,
        IOError,
    ),
)

# For embedding generation
embedding_retry = create_retry_decorator(
    max_attempts=3,
    min_wait=1.0,
    max_wait=15.0,
    retry_exceptions=(
        ConnectionError,
        TimeoutError,
        IOError,
    ),
)


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    retry_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        IOError,
    ),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to add retry logic to a function.

    Usage:
        @with_retry(max_attempts=3)
        async def my_api_call():
            ...
    """
    return create_retry_decorator(
        max_attempts=max_attempts,
        min_wait=min_wait,
        max_wait=max_wait,
        retry_exceptions=retry_exceptions,
    )
