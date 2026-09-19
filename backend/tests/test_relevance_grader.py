"""
Unit tests for Relevance Grader agent.

The grader uses ``llm.with_structured_output(...)``; the stub below answers that
call deterministically so the real parsing/return path is covered.
"""

import pytest

from backend.core.agents import relevance_grader
from backend.core.agents.relevance_grader import GradeChunk, grade_chunk_relevance
from backend.tests.llm_stubs import StructuredFakeChatModel, fake_llm_factory

pytestmark = pytest.mark.unit


class TestRelevanceGrader:
    """Tests for the relevance grader agent."""

    def test_grade_relevant_chunk(self, monkeypatch):
        """Should return True when the model grades the chunk as relevant."""
        monkeypatch.setattr(
            relevance_grader,
            "get_llm",
            fake_llm_factory(StructuredFakeChatModel(binary_score="yes")),
        )

        result = grade_chunk_relevance(
            "What is LangGraph?", "LangGraph is a library for building apps"
        )

        assert result is True

    def test_grade_irrelevant_chunk(self, monkeypatch):
        """Should return False when the model grades the chunk as irrelevant."""
        monkeypatch.setattr(
            relevance_grader,
            "get_llm",
            fake_llm_factory(StructuredFakeChatModel(binary_score="no")),
        )

        result = grade_chunk_relevance("What is LangGraph?", "Python is a programming language")

        assert result is False

    def test_grade_chunk_model(self):
        """Test GradeChunk model validation."""
        grade = GradeChunk(binary_score="yes")
        assert grade.binary_score == "yes"

        grade = GradeChunk(binary_score="no")
        assert grade.binary_score == "no"
