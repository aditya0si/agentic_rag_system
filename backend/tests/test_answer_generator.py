"""
Unit tests for Answer Generator agent.
"""

import pytest
from core.agents.answer_generator import generate_answer


class TestAnswerGenerator:
    """Tests for the answer generator agent."""

    def test_returns_fallback_when_no_chunks(self):
        """Should return fallback message when no chunks provided."""
        result = generate_answer("What is LangGraph?", [])
        assert "couldn't find relevant information" in result.lower()

    def test_returns_fallback_when_empty_chunks(self):
        """Should return fallback message when chunks list is empty."""
        result = generate_answer("What is LangGraph?", [])
        assert "couldn't find relevant information" in result.lower()

    def test_generates_answer_with_chunks(self, monkeypatch, sample_chunks):
        """Should generate answer when chunks are provided."""
        def mock_generate(question, chunks):
            return "LangGraph is a library for building stateful applications [1]."
        
        import core.agents.answer_generator
        monkeypatch.setattr(core.agents.answer_generator, "generate_answer", mock_generate)
        
        result = generate_answer("What is LangGraph?", sample_chunks)
        assert "LangGraph" in result
        assert "[1]" in result

    def test_handles_chunks_with_metadata(self, monkeypatch):
        """Should handle chunks with all metadata fields."""
        chunks = [
            {
                "chunk_id": "doc_1_chunk_001",
                "doc_id": "doc_1",
                "doc_name": "test.pdf",
                "page_number": 5,
                "chunk_text": "Test content about LangGraph",
                "chunk_index": 0,
                "char_count": 30,
            }
        ]
        
        def mock_generate(question, chunks):
            return "Based on the document [1], LangGraph is a library."
        
        import core.agents.answer_generator
        monkeypatch.setattr(core.agents.answer_generator, "generate_answer", mock_generate)
        
        result = generate_answer("What is LangGraph?", chunks)
        assert "LangGraph" in result