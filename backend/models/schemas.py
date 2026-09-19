"""
Pydantic request/response models for the API.

These define the contract between frontend and backend.
"""

from pydantic import BaseModel, Field

# =============================================================================
# Upload Models
# =============================================================================


class UploadResponse(BaseModel):
    """Response returned after successful document ingestion."""

    doc_id: str = Field(..., description="Unique identifier for the ingested document")
    filename: str = Field(..., description="Original filename")
    chunks_created: int = Field(..., description="Number of chunks created from the document")
    status: str = Field(default="indexed", description="Ingestion status")


# =============================================================================
# Ask Models
# =============================================================================


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    session_id: str = Field(..., description="Session identifier for chat history")
    question: str = Field(..., description="User's natural-language question")
    doc_ids: list[str] | None = Field(
        default=None,
        description="Optional list of doc_ids to scope the search to specific documents",
    )


class Citation(BaseModel):
    """A single citation linking an answer claim to its source."""

    doc_id: str = Field(..., description="Document the citation comes from")
    doc_name: str = Field(default="", description="Original document filename")
    page: int = Field(..., description="Page number in the source document")
    chunk_text: str = Field(..., description="Exact text from the source chunk")


class AskResponse(BaseModel):
    """Response returned from the /ask endpoint."""

    answer: str = Field(..., description="Generated answer grounded in the source documents")
    citations: list[Citation] = Field(
        default_factory=list, description="Citations linking answer claims to source chunks"
    )
    agent_trace: list[str] = Field(
        default_factory=list,
        description="Ordered list of agent names that participated in answering",
    )
    hallucination_warning: str | None = Field(
        default=None, description="Warning if the hallucination checker flagged the answer"
    )


# =============================================================================
# Error Models
# =============================================================================


class ErrorResponse(BaseModel):
    """Structured error response."""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
