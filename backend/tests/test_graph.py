"""
Unit tests for Agentic RAG Graph pipeline.
"""

import pytest
from core.agents.graph import run_agentic_rag, AgentState


class TestAgenticRAGGraph:
    """Tests for the LangGraph agentic RAG pipeline."""

    def test_run_agentic_rag_returns_complete_state(self, monkeypatch, sample_chat_history):
        """Should return complete state with all fields."""
        def mock_run(question, chat_history, doc_ids):
            return {
                "question": question,
                "doc_ids": doc_ids,
                "chat_history": chat_history,
                "rewritten_query": "standalone query",
                "retrieved_chunks": [{"chunk_text": "test"}],
                "relevant_chunks": [{"chunk_text": "test"}],
                "answer": "Mocked answer",
                "agent_trace": [
                    "query_rewriter",
                    "retriever", 
                    "relevance_grader",
                    "answer_generator",
                    "hallucination_checker"
                ],
                "hallucination_warning": None
            }
        
        import core.agents.graph
        monkeypatch.setattr(core.agents.graph, "run_agentic_rag", mock_run)
        
        result = run_agentic_rag(
            "What is LangGraph?",
            chat_history=sample_chat_history,
            doc_ids=["doc_1"]
        )
        
        assert "question" in result
        assert "answer" in result
        assert "agent_trace" in result
        assert "hallucination_warning" in result
        assert len(result["agent_trace"]) == 5

    def test_agent_trace_contains_all_nodes(self, monkeypatch):
        """Agent trace should contain all 5 nodes in order."""
        def mock_run(question, chat_history, doc_ids):
            return {
                "agent_trace": [
                    "query_rewriter",
                    "retriever",
                    "relevance_grader", 
                    "answer_generator",
                    "hallucination_checker"
                ],
                "hallucination_warning": None
            }
        
        import core.agents.graph
        monkeypatch.setattr(core.agents.graph, "run_agentic_rag", mock_run)
        
        result = run_agentic_rag("test", [], ["doc_1"])
        trace = result["agent_trace"]
        
        assert trace[0] == "query_rewriter"
        assert trace[1] == "retriever"
        assert trace[2] == "relevance_grader"
        assert trace[3] == "answer_generator"
        assert trace[4] == "hallucination_checker"

    def test_hallucination_warning_when_detected(self, monkeypatch):
        """Should include warning when hallucination detected."""
        def mock_run(question, chat_history, doc_ids):
            return {
                "answer": "Some answer",
                "hallucination_warning": "WARNING: Some claims may not be fully supported by the source documents"
            }
        
        import core.agents.graph
        monkeypatch.setattr(core.agents.graph, "run_agentic_rag", mock_run)
        
        result = run_agentic_rag("test", [], ["doc_1"])
        assert result["hallucination_warning"] is not None
        assert "WARNING" in result["hallucination_warning"]

    def test_no_warning_when_grounded(self, monkeypatch):
        """Should not include warning when answer is grounded."""
        def mock_run(question, chat_history, doc_ids):
            return {
                "answer": "Grounded answer",
                "hallucination_warning": None
            }
        
        import core.agents.graph
        monkeypatch.setattr(core.agents.graph, "run_agentic_rag", mock_run)
        
        result = run_agentic_rag("test", [], ["doc_1"])
        assert result["hallucination_warning"] is None

    def test_initial_state_has_all_fields(self):
        """AgentState TypedDict should have all required fields."""
        state: AgentState = {
            "question": "test",
            "doc_ids": ["doc_1"],
            "chat_history": [{"role": "user", "content": "hi"}],
            "rewritten_query": "",
            "retrieved_chunks": [],
            "relevant_chunks": [],
            "answer": "",
            "agent_trace": [],
            "hallucination_warning": None
        }
        assert "question" in state
        assert "hallucination_warning" in state