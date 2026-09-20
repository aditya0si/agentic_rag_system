"""
Unit tests for Query Rewriter agent.

The LLM is stubbed so the real history-formatting and chain code in
``rewrite_query`` runs deterministically.
"""

import pytest

from backend.core.agents import query_rewriter
from backend.core.agents.query_rewriter import rewrite_query
from backend.tests.llm_stubs import RecordingFakeChatModel, fake_llm_factory

pytestmark = pytest.mark.unit


class TestQueryRewriter:
    """Tests for the query rewriter agent."""

    def test_returns_original_when_no_history(self):
        """Should return original question when no chat history."""
        question = "What is LangGraph?"
        result = rewrite_query(question, [])
        assert result == question

    def test_rewrites_with_history(self, monkeypatch, sample_chat_history):
        """Should send the formatted history to the LLM and return its rewrite."""
        model = RecordingFakeChatModel(response="standalone rewritten query")
        monkeypatch.setattr(query_rewriter, "get_llm", fake_llm_factory(model))

        result = rewrite_query("What about its features?", sample_chat_history)

        assert result == "standalone rewritten query"
        prompt = "\n".join(model.rendered_prompts())
        assert "Tell me about LangGraph" in prompt
        assert "What about its features?" in prompt


class TestQueryRewriterWithMocks:
    """Tests with explicit mocking of the LLM boundary."""

    def test_rewrite_query_standalone(self, monkeypatch):
        """Test query rewriting with a stubbed LLM."""
        model = RecordingFakeChatModel(response="standalone rewritten query")
        monkeypatch.setattr(query_rewriter, "get_llm", fake_llm_factory(model))

        result = rewrite_query("follow up question", [{"role": "user", "content": "previous"}])

        assert result == "standalone rewritten query"
