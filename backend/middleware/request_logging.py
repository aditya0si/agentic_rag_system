"""
Request logging middleware — logs all incoming requests and outgoing responses.
"""

import time
import uuid
from collections.abc import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all HTTP requests with timing and metadata.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID for tracing
        request_id = str(uuid.uuid4())

        # Start timer
        start_time = time.time()

        # Log request
        logger.info(
            "http_request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            client_host=request.client.host if request.client else None,
        )

        # Add request ID to request state for downstream use
        request.state.request_id = request_id

        try:
            # Process request
            response: Response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log response
            logger.info(
                "http_request_completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
            )

            # Add request ID to response headers for tracing
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as exc:
            # Calculate duration
            duration = time.time() - start_time

            # Log error
            logger.error(
                "http_request_failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration * 1000, 2),
                error=str(exc),
                exc_info=True,
            )
            raise
