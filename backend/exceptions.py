"""
Custom exception hierarchy for the Agentic RAG Research Assistant.

Provides structured error handling with error codes for API responses.
"""

from typing import Any


class RAGError(Exception):
    """Base exception for all RAG pipeline errors."""

    def __init__(
        self, message: str, error_code: str = "RAG_ERROR", details: dict[str, Any] | None = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ConfigurationError(RAGError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, error_code="CONFIGURATION_ERROR", details=details)


class IngestionError(RAGError):
    """Raised when document ingestion fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, error_code="INGESTION_ERROR", details=details)


class UnsupportedFileTypeError(IngestionError):
    """Raised when an unsupported file type is uploaded."""

    def __init__(self, file_type: str):
        super().__init__(f"Unsupported file type: {file_type}", details={"file_type": file_type})


class FileTooLargeError(IngestionError):
    """Raised when uploaded file exceeds size limit."""

    def __init__(self, max_size_mb: int):
        super().__init__(
            f"File exceeds maximum size limit of {max_size_mb}MB",
            details={"max_size_mb": max_size_mb},
        )


class ParsingError(IngestionError):
    """Raised when document parsing fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"Failed to parse document: {message}", details=details)


class RetrievalError(RAGError):
    """Raised when document retrieval fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, error_code="RETRIEVAL_ERROR", details=details)


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, error_code="EMBEDDING_ERROR", details=details)


class VectorStoreError(RAGError):
    """Raised when vector store operations fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, error_code="VECTOR_STORE_ERROR", details=details)


class LLMError(RAGError):
    """Raised when LLM operations fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, error_code="LLM_ERROR", details=details)


class QueryRewriteError(LLMError):
    """Raised when query rewriting fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"Query rewrite failed: {message}", details=details)


class GenerationError(LLMError):
    """Raised when answer generation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"Answer generation failed: {message}", details=details)


class GradingError(LLMError):
    """Raised when relevance grading fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"Relevance grading failed: {message}", details=details)


class HallucinationCheckError(LLMError):
    """Raised when hallucination checking fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"Hallucination check failed: {message}", details=details)


class SessionError(RAGError):
    """Raised when session management fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, error_code="SESSION_ERROR", details=details)


class ValidationError(RAGError):
    """Raised when request validation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, error_code="VALIDATION_ERROR", details=details)
