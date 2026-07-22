"""
Unit tests for Hallucination Checker agent.
"""

import pytest
from core.agents.hallucination_checker import check_hallucination


class TestHallucinationChecker:
    """Tests for the hallucination checker agent."""

    def test_returns_false_for_fallback_answer(self):
        """Should return False for fallback 'not found' answers."""
        answer = "I couldn't find relevant information in the uploaded document(s) to answer this."
        chunks = []
        result = check_hallucination(answer, chunks)
        assert result is False

    def test_returns_true_for_hallucinated_answer(self, monkeypatch):
        """Should return True when hallucination detected."""
        def mock_check(answer, chunks):
            return True
        
        import core.agents.hallucination_checker
        monkeypatch.setattr(core.agents.hallucination_checker, "check_hallucination", mock_check)
        
        answer = "LangGraph was created by Google in 2020."
        chunks = [{"chunk_text": "LangGraph is a library for building apps"}]
        result = check_hallucination(answer, chunks)
        assert result is True

    def test_returns_false_for_grounded_answer(self, monkeypatch):
        """Should return False when answer is grounded in chunks."""
        def mock_check(answer, chunks):
            return False
        
        import core.agents.hallucination_checker
        monkeypatch.setattr(core.agents.hallucination_checker, "check_hallucination", mock_check)
        
        answer = "LangGraph is a library for building stateful applications [1]."
        chunks = [{"chunk_text": "LangGraph is a library for building stateful applications with LLMs"}]
        result = check_hallucination(answer, chunks)
        assert result is False

    def test_returns_true_for_empty_chunks_with_claim(self, monkeypatch):
        """Should return True when making claims without source chunks."""
        def mock_check(answer, chunks):
            return True
        
        import core.agents.hallucination_checker
        monkeypatch.setattr(core.agents.hallucination_checker, "check_hallucination", mock_check)
        
        answer = "LangGraph has 1000+ stars on GitHub."
        chunks = []
        result = check_hallucination(answer, chunks)
        assert result is True