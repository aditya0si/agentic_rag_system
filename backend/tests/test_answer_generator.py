"""
Unit tests for Answer Generator agent.

The hosted LLM is replaced with a deterministic in-memory chat model, so the
real prompt construction, chain wiring and output parsing in
``generate_answer`` are exercised without network access.
"""

import pytest

from backend.core.agents import answer_generator
from backend.core.agents.answer_generator import generate_answer
from backend.tests.llm_stubs import RecordingFakeChatModel, fake_llm_factory

pytestmark = pytest.mark.unit


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
        """Should run the prompt/chain and return the model answer."""
        model = RecordingFakeChatModel(
            response="LangGraph is a library for building stateful applications [1]."
        )
        monkeypatch.setattr(answer_generator, "get_llm", fake_llm_factory(model))

        result = generate_answer("What is LangGraph?", sample_chunks)

        assert "LangGraph" in result
        assert "[1]" in result
        prompt = "\n".join(model.rendered_prompts())
        assert "What is LangGraph?" in prompt
        assert sample_chunks[0]["chunk_text"] in prompt

    def test_handles_chunks_with_metadata(self, monkeypatch):
        """Should include every chunk's source metadata in the prompt."""
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
        model = RecordingFakeChatModel(
            response="Based on the document [1], LangGraph is a library."
        )
        monkeypatch.setattr(answer_generator, "get_llm", fake_llm_factory(model))

        result = generate_answer("What is LangGraph?", chunks)

        assert "LangGraph" in result
        prompt = "\n".join(model.rendered_prompts())
        assert "test.pdf" in prompt
        assert "Test content about LangGraph" in prompt
