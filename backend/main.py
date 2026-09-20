"""
Agentic RAG Research Assistant — FastAPI Application Entrypoint

Registers all routers, middleware, and configures the application.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import CORS_ORIGINS, validate_config
from .logging_config import configure_logging
from .middleware.rate_limiter import RateLimitMiddleware
from .middleware.request_logging import RequestLoggingMiddleware
from .routers import ask, health, stream, upload
from .routers import metrics as metrics_router

# =============================================================================
# Logging — configure structured JSON logging before anything else
# =============================================================================
configure_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    json_output=os.getenv("LOG_JSON", "true").lower() == "true",
)

# =============================================================================
# Application Factory
# =============================================================================

app = FastAPI(
    title="Agentic RAG Research Assistant",
    description=(
        "A multi-agent RAG pipeline that lets users upload documents and ask "
        "natural-language questions, receiving source-grounded, cited answers."
    ),
    version="0.2.0",
)

# ---------------------------------------------------------------------------
# Middleware (order matters: last added = outermost)
# ---------------------------------------------------------------------------

# CORS — explicitly allow the deployed web client. Configure additional
# origins through CORS_ORIGINS rather than exposing credentialed APIs to '*'.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting — 100 requests/minute per client by default
app.add_middleware(
    RateLimitMiddleware,
    max_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
    window_seconds=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
)

# Request/response structured logging with X-Request-ID tracing
app.add_middleware(RequestLoggingMiddleware)

# ---------------------------------------------------------------------------
# Register Routers
# ---------------------------------------------------------------------------
app.include_router(health.router, tags=["Health"])
app.include_router(upload.router, tags=["Document Upload"])
app.include_router(ask.router, tags=["Question Answering"])
app.include_router(metrics_router.router, tags=["Observability"])
app.include_router(stream.router, tags=["Streaming"])

# ---------------------------------------------------------------------------
# Startup Event — validate configuration
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    warnings = validate_config()
    if warnings:
        for w in warnings:
            print(f"[WARNING] CONFIG WARNING: {w}")
    else:
        print("[INFO] Configuration validated successfully.")
