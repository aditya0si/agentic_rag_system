"""
Integration tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from main import app


class TestEndpoints:
    """Integration tests for API endpoints."""

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
        # Mock the ingestion pipeline
        def mock_extract_text(file_path):
            return [{"text": "Test content", "page_number": 1}]
        
        def mock_chunk_document(pages, doc_id, doc_name, chunk_size, chunk_overlap):
            return [{
                "chunk_id": f"{doc_id}_chunk_000",
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page_number": 1,
                "chunk_text": "Test content",
                "chunk_index": 0,
                "char_count": 12,
            }]
        
        def mock_add_chunks(chunks):
            pass
        
        def mock_add_document_to_session(session_id, doc_id):
            pass
        
        import core.ingestion
        import core.vector_store
        import core.memory
        
        monkeypatch.setattr(core.ingestion, "extract_text", mock_extract_text)
        monkeypatch.setattr(core.ingestion, "chunk_document", mock_chunk_document)
        monkeypatch.setattr(core.vector_store.VectorStoreManager, "add_chunks", mock_add_chunks)
        monkeypatch.setattr(core.memory, "add_document_to_session", mock_add_document_to_session)
        
        files = {"file": ("test.txt", b"Test content", "text/plain")}
        response = client.post("/upload", files=files, data={"session_id": "test_session"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "indexed"
        assert "doc_id" in data
        assert data["chunks_created"] >= 0

    def test_ask_without_documents(self, client):
        """Should return helpful message when no documents uploaded."""
        response = client.post("/ask", json={
            "session_id": "empty_session",
            "question": "What is LangGraph?"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "upload at least one document" in data["answer"].lower()

    def test_ask_with_documents(self, client, monkeypatch):
        """Should return answer with citations when documents exist."""
        def mock_get_session_documents(session_id):
            return ["doc_test"]
        
        def mock_get_chat_history(session_id):
            return []
        
        def mock_add_message_to_history(session_id, role, content):
            pass
        
        def mock_run_agentic_rag(question, chat_history, doc_ids):
            return {
                "answer": "LangGraph is a library for building stateful apps [1].",
                "relevant_chunks": [{
                    "doc_id": "doc_test",
                    "doc_name": "test.txt",
                    "page_number": 1,
                    "chunk_text": "LangGraph is a library for building stateful apps",
                }],
                "agent_trace": [
                    "query_rewriter",
                    "retriever",
                    "relevance_grader",
                    "answer_generator",
                    "hallucination_checker"
                ],
                "hallucination_warning": None
            }
        
        import core.memory
        import core.agents.graph
        
        monkeypatch.setattr(core.memory, "get_session_documents", mock_get_session_documents)
        monkeypatch.setattr(core.memory, "get_chat_history", mock_get_chat_history)
        monkeypatch.setattr(core.memory, "add_message_to_history", mock_add_message_to_history)
        monkeypatch.setattr(core.agents.graph, "run_agentic_rag", mock_run_agentic_rag)
        
        response = client.post("/ask", json={
            "session_id": "test_session",
            "question": "What is LangGraph?"
        })
        
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
            "hallucination_checker"
        ]

    def test_ask_with_hallucination_warning(self, client, monkeypatch):
        """Should include hallucination warning when detected."""
        def mock_get_session_documents(session_id):
            return ["doc_test"]
        
        def mock_get_chat_history(session_id):
            return []
        
        def mock_add_message_to_history(session_id, role, content):
            pass
        
        def mock_run_agentic_rag(question, chat_history, doc_ids):
            return {
                "answer": "Hallucinated answer",
                "relevant_chunks": [],
                "agent_trace": [
                    "query_rewriter",
                    "retriever",
                    "relevance_grader",
                    "answer_generator",
                    "hallucination_checker"
                ],
                "hallucination_warning": "WARNING: Some claims may not be fully supported by the source documents"
            }
        
        import core.memory
        import core.agents.graph
        
        monkeypatch.setattr(core.memory, "get_session_documents", mock_get_session_documents)
        monkeypatch.setattr(core.memory, "get_chat_history", mock_get_chat_history)
        monkeypatch.setattr(core.memory, "add_message_to_history", mock_add_message_to_history)
        monkeypatch.setattr(core.agents.graph, "run_agentic_rag", mock_run_agentic_rag)
        
        response = client.post("/ask", json={
            "session_id": "test_session",
            "question": "What is LangGraph?"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["hallucination_warning"] is not None
        assert "WARNING" in data["hallucination_warning"]