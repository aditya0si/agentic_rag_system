"""
Structured logging configuration using structlog.

Provides JSON-formatted logs with consistent fields for observability.
"""

import logging
import sys
from typing import Any, cast

import structlog


def configure_logging(
    log_level: str = "INFO",
    json_output: bool = True,
) -> None:
    """
    Configure structlog for structured logging.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        json_output: If True, output JSON; if False, output human-readable
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Shared processors for all loggers
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        # JSON output for production / log aggregation
        processors = shared_processors + [structlog.processors.JSONRenderer()]
    else:
        # Human-readable output for development
        processors = shared_processors + [structlog.dev.ConsoleRenderer(colors=True)]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger
    """
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
