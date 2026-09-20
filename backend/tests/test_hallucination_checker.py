"""
Unit tests for Hallucination Checker agent.

The LLM boundary is stubbed with a deterministic structured-output model so the
real chunk-formatting and verdict parsing code is covered.
"""

from typing import Any

import pytest

from backend.core.agents import hallucination_checker
from backend.core.agents.hallucination_checker import check_hallucination
from backend.tests.llm_stubs import StructuredFakeChatModel, fake_llm_factory

pytestmark = pytest.mark.unit


class TestHallucinationChecker:
    """Tests for the hallucination checker agent."""

    def test_returns_false_for_fallback_answer(self):
        """Should return False for fallback 'not found' answers."""
        answer = "I couldn't find relevant information in the uploaded document(s) to answer this."
        chunks: list[dict[str, Any]] = []
        result = check_hallucination(answer, chunks)
        assert result is False

    def test_returns_true_for_hallucinated_answer(self, monkeypatch):
        """Should return True when the model flags unsupported claims."""
        monkeypatch.setattr(
            hallucination_checker,
            "get_llm",
            fake_llm_factory(StructuredFakeChatModel(binary_score="yes")),
        )

        answer = "LangGraph was created by Google in 2020."
        chunks = [{"chunk_text": "LangGraph is a library for building apps"}]
        result = check_hallucination(answer, chunks)
        assert result is True

    def test_returns_false_for_grounded_answer(self, monkeypatch):
        """Should return False when the model finds the answer grounded."""
        monkeypatch.setattr(
            hallucination_checker,
            "get_llm",
            fake_llm_factory(StructuredFakeChatModel(binary_score="no")),
        )

        answer = "LangGraph is a library for building stateful applications [1]."
        chunks = [
            {"chunk_text": "LangGraph is a library for building stateful applications with LLMs"}
        ]
        result = check_hallucination(answer, chunks)
        assert result is False

    def test_returns_true_for_empty_chunks_with_claim(self):
        """Should return True when making claims without source chunks."""
        answer = "LangGraph has 1000+ stars on GitHub."
        chunks: list[dict[str, Any]] = []
        result = check_hallucination(answer, chunks)
        assert result is True
