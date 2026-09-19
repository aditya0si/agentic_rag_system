"""
Rate limiting middleware — prevents abuse by limiting requests per client.
"""

import time
from collections import defaultdict
from collections.abc import Callable

import structlog
from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter using token bucket algorithm.
    Limits requests per client IP address.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_requests: int = 100,
        window_seconds: int = 60,
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # Store: {client_ip: [(timestamp1), (timestamp2), ...]}
        self.requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path == "/health":
            health_response: Response = await call_next(request)
            return health_response

        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # Clean old requests outside the window
        self.requests[client_ip] = [
            req_time
            for req_time in self.requests[client_ip]
            if current_time - req_time < self.window_seconds
        ]

        # Check if limit exceeded
        if len(self.requests[client_ip]) >= self.max_requests:
            logger.warning(
                "rate_limit_exceeded",
                client_ip=client_ip,
                path=request.url.path,
                requests_count=len(self.requests[client_ip]),
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds}s.",
            )

        # Record this request
        self.requests[client_ip].append(current_time)

        # Process request
        response: Response = await call_next(request)

        # Add rate limit headers
        remaining = self.max_requests - len(self.requests[client_ip])
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.window_seconds))

        return response
