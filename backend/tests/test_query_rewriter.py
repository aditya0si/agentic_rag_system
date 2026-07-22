"""
Unit tests for Query Rewriter agent.
"""

import pytest
from core.agents.query_rewriter import rewrite_query


class TestQueryRewriter:
    """Tests for the query rewriter agent."""

    def test_returns_original_when_no_history(self):
        """Should return original question when no chat history."""
        question = "What is LangGraph?"
        result = rewrite_query(question, [])
        assert result == question

    def test_returns_original_when_empty_history(self):
        """Should return original question when chat history is empty list."""
        question = "What is LangGraph?"
        result = rewrite_query(question, [{}])
        assert result == question

    def test_rewrites_with_history(self, mock_chain):
        """Should rewrite question when chat history exists."""
        question = "What about its features?"
        history = [
            {"role": "user", "content": "Tell me about LangGraph"},
            {"role": "assistant", "content": "LangGraph is a library for building stateful apps"},
        ]
        result = rewrite_query(question, history)
        assert "standalone" in result.lower() or "langgraph" in result.lower()


class TestQueryRewriterWithMocks:
    """Tests with explicit mocking."""

    def test_rewrite_query_standalone(self, monkeypatch):
        """Test query rewriting with mocked LLM."""
        def mock_rewrite(question, chat_history):
            return "standalone rewritten query"
        
        import core.agents.query_rewriter
        monkeypatch.setattr(core.agents.query_rewriter, "rewrite_query", mock_rewrite)
        
        result = rewrite_query("follow up question", [{"role": "user", "content": "previous"}])
        assert result == "standalone rewritten query"