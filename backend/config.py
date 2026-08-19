"""
Configuration module — loads environment variables from .env file.

All application settings are centralized here. Other modules import
from this file rather than reading os.environ directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the backend directory
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path)


# =============================================================================
# LLM Provider Configuration
# =============================================================================

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "google")  # "openai" or "google"

# OpenAI
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Google Gemini
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL: str = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")


# =============================================================================
# Embedding Configuration
# =============================================================================

EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")  # "openai" or "huggingface"
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
HUGGINGFACE_EMBEDDING_MODEL: str = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


# =============================================================================
# LangSmith Observability
# =============================================================================

LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "agentic-rag-assistant")


# =============================================================================
# Application Settings
# =============================================================================

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", "20971520"))  # 20MB
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "6"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# Web deployment. Keep the default deliberately narrow: wildcard origins cannot
# safely be used together with credentialed browser requests in production.
CORS_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
    if origin.strip()
]
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").lower()


# =============================================================================
# Validation
# =============================================================================

def validate_config() -> list[str]:
    """
    Validates that required configuration is present.
    Returns a list of warnings (empty if all is well).
    """
    warnings = []

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        warnings.append("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set.")
    if LLM_PROVIDER == "google" and not GOOGLE_API_KEY:
        warnings.append("LLM_PROVIDER is 'google' but GOOGLE_API_KEY is not set.")
    if EMBEDDING_PROVIDER == "openai" and not OPENAI_API_KEY:
        warnings.append("EMBEDDING_PROVIDER is 'openai' but OPENAI_API_KEY is not set.")
    if "*" in CORS_ORIGINS and ENVIRONMENT == "production":
        warnings.append("CORS_ORIGINS must not contain '*' in production.")

    return warnings
