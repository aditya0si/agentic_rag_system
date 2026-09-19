"""
Endpoint tests for the FastAPI app.

These use FastAPI's ``TestClient`` with the pipeline boundaries (document
ingestion, vector store, session memory, agentic graph) stubbed out, so they run
without network access, model downloads or API credentials.

Note on patch targets: the routers import their dependencies by name
(``from ..core.agents.graph import run_agentic_rag``), so a test must patch the
name *where it is looked up* - ``backend.routers.ask`` - rather than the module
it was defined in. Patching the defining module (as this file used to do) left
the real implementation in place.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.routers import ask as ask_router
from backend.routers import upload as upload_router

pytestmark = pytest.mark.unit


class TestEndpoints:
    """Endpoint behaviour with stubbed pipeline boundaries."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Health endpoint should return OK."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_upload_unsupported_file_type(self, client):
        """Should reject unsupported file types."""
        files = {"file": ("test.py", b"print('hello')", "text/plain")}
        response = client.post("/upload", files=files, data={"session_id": "test_session"})

        assert response.status_code == 422
        assert response.json()["error"] == "unsupported_file_type"

    def test_upload_text_file(self, client, monkeypatch):
        """Should accept and process text files."""
        ingested: dict[str, Any] = {}

        def fake_extract_text(file_path):
            return [{"text": "Test content", "page_number": 1}]

        def fake_chunk_document(pages, doc_id, doc_name, chunk_size=500, chunk_overlap=50):
            return [
                {
                    "chunk_id": f"{doc_id}_chunk_000",
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page_number": 1,
                    "chunk_text": "Test content",
                    "chunk_index": 0,
                    "char_count": 12,
                }
            ]

        class FakeVectorStoreManager:
            def add_chunks(self, chunks):
                ingested["chunks"] = chunks

        def fake_add_document_to_session(session_id, doc_id):
            ingested["session"] = (session_id, doc_id)

        monkeypatch.setattr(upload_router, "extract_text", fake_extract_text)
        monkeypatch.setattr(upload_router, "chunk_document", fake_chunk_document)
        monkeypatch.setattr(upload_router, "VectorStoreManager", FakeVectorStoreManager)
        monkeypatch.setattr(upload_router, "add_document_to_session", fake_add_document_to_session)

        files = {"file": ("test.txt", b"Test content", "text/plain")}
        response = client.post("/upload", files=files, data={"session_id": "test_session"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "indexed"
        assert "doc_id" in data
        assert data["chunks_created"] == 1
        # The stubs prove the ingestion pipeline was reached with the parsed pages.
        assert ingested["chunks"][0]["chunk_text"] == "Test content"
        assert ingested["session"] == ("test_session", data["doc_id"])

    def test_ask_without_documents(self, client, monkeypatch):
        """Should return helpful message when no documents uploaded."""
        monkeypatch.setattr(ask_router, "get_session_documents", lambda session_id: [])

        response = client.post(
            "/ask", json={"session_id": "empty_session", "question": "What is LangGraph?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "upload at least one document" in data["answer"].lower()

    def test_ask_with_documents(self, client, monkeypatch):
        """Should return answer with citations when documents exist."""

        def mock_run_agentic_rag(question, chat_history, doc_ids):
            return {
                "answer": "LangGraph is a library for building stateful apps [1].",
                "relevant_chunks": [
                    {
                        "doc_id": "doc_test",
                        "doc_name": "test.txt",
                        "page_number": 1,
                        "chunk_text": "LangGraph is a library for building stateful apps",
                    }
                ],
                "agent_trace": [
                    "query_rewriter",
                    "retriever",
                    "relevance_grader",
                    "answer_generator",
                    "hallucination_checker",
                ],
                "hallucination_warning": None,
            }

        monkeypatch.setattr(ask_router, "get_session_documents", lambda session_id: ["doc_test"])
        monkeypatch.setattr(ask_router, "get_chat_history", lambda session_id: [])
        monkeypatch.setattr(ask_router, "add_message_to_history", lambda *args, **kwargs: None)
        monkeypatch.setattr(ask_router, "run_agentic_rag", mock_run_agentic_rag)

        response = client.post(
            "/ask", json={"session_id": "test_session", "question": "What is LangGraph?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "LangGraph" in data["answer"]
        assert len(data["citations"]) > 0
        assert data["citations"][0]["doc_id"] == "doc_test"
        assert data["agent_trace"] == [
            "query_rewriter",
            "retriever",
            "relevance_grader",
            "answer_generator",
            "hallucination_checker",
        ]

    def test_ask_with_hallucination_warning(self, client, monkeypatch):
        """Should include hallucination warning when detected."""

        def mock_run_agentic_rag(question, chat_history, doc_ids):
            return {
                "answer": "Hallucinated answer",
                "relevant_chunks": [],
                "agent_trace": [
                    "query_rewriter",
                    "retriever",
                    "relevance_grader",
                    "answer_generator",
                    "hallucination_checker",
                ],
                "hallucination_warning": (
                    "WARNING: Some claims may not be fully supported by the source documents"
                ),
            }

        monkeypatch.setattr(ask_router, "get_session_documents", lambda session_id: ["doc_test"])
        monkeypatch.setattr(ask_router, "get_chat_history", lambda session_id: [])
        monkeypatch.setattr(ask_router, "add_message_to_history", lambda *args, **kwargs: None)
        monkeypatch.setattr(ask_router, "run_agentic_rag", mock_run_agentic_rag)

        response = client.post(
            "/ask", json={"session_id": "test_session", "question": "What is LangGraph?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["hallucination_warning"] is not None
        assert "WARNING" in data["hallucination_warning"]
