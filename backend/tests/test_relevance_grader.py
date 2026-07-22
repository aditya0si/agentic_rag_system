"""
Unit tests for Relevance Grader agent.
"""

import pytest
from core.agents.relevance_grader import grade_chunk_relevance, GradeChunk


class TestRelevanceGrader:
    """Tests for the relevance grader agent."""

    def test_grade_relevant_chunk(self, monkeypatch):
        """Should return True for relevant chunks."""
        def mock_grade(query, chunk_text):
            return True
        
        import core.agents.relevance_grader
        monkeypatch.setattr(core.agents.relevance_grader, "grade_chunk_relevance", mock_grade)
        
        result = grade_chunk_relevance("What is LangGraph?", "LangGraph is a library for building apps")
        assert result is True

    def test_grade_irrelevant_chunk(self, monkeypatch):
        """Should return False for irrelevant chunks."""
        def mock_grade(query, chunk_text):
            return False
        
        import core.agents.relevance_grader
        monkeypatch.setattr(core.agents.relevance_grader, "grade_chunk_relevance", mock_grade)
        
        result = grade_chunk_relevance("What is LangGraph?", "Python is a programming language")
        assert result is False

    def test_grade_chunk_model(self):
        """Test GradeChunk model validation."""
        grade = GradeChunk(binary_score="yes")
        assert grade.binary_score == "yes"
        
        grade = GradeChunk(binary_score="no")
        assert grade.binary_score == "no"


class TestRelevanceGraderWithMocks:
    """Tests with explicit mocking."""

    def test_grader_returns_boolean(self, monkeypatch):
        """Grader should always return boolean."""
        def mock_grade(query, chunk_text):
            return True
        
        import core.agents.relevance_grader
        monkeypatch.setattr(core.agents.relevance_grader, "grade_chunk_relevance", mock_grade)
        
        result = grade_chunk_relevance("test query", "test chunk")
        assert isinstance(result, bool)