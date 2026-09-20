"""
Unit tests for the Agentic RAG Graph pipeline.

These tests invoke the real compiled LangGraph pipeline (``run_agentic_rag``)
and only stub its agent boundaries (retrieval, grading, generation,
hallucination check). That way the graph wiring, state merging and agent-trace
bookkeeping are actually verified. The previous version of this file
monkeypatched ``run_agentic_rag`` itself and then asserted on the stub's return
value, so it could not detect a broken pipeline.
"""

from typing import Any

import pytest

from backend.core.agents import graph
from backend.core.agents.graph import AgentState, run_agentic_rag

pytestmark = pytest.mark.unit


def _chunk(index: int, text: str) -> dict[str, Any]:
    return {
        "chunk_id": f"doc_test_chunk_{index:03d}",
        "doc_id": "doc_test",
        "doc_name": "test.txt",
        "page_number": 1,
        "chunk_text": text,
        "chunk_index": index,
        "char_count": len(text),
    }


@pytest.fixture
def stub_agents(monkeypatch):
    """Replace the pipeline's external boundaries with deterministic stubs."""
    monkeypatch.setattr(
        graph, "rewrite_query", lambda question, chat_history: f"{question} (rewritten)"
    )
    monkeypatch.setattr(
        graph,
        "retrieve_chunks",
        lambda query, doc_ids=None, top_k=None: [
            _chunk(0, "LangGraph is a library for building stateful applications."),
            _chunk(1, "Unrelated sentence about the weather."),
        ],
    )
    monkeypatch.setattr(graph, "rerank_chunks", lambda query, chunks, top_k=None: chunks)
    monkeypatch.setattr(
        graph,
        "grade_chunk_relevance",
        lambda query, chunk_text: "LangGraph" in chunk_text,
    )
    monkeypatch.setattr(graph, "generate_answer", lambda question, chunks: "Mocked answer [1]")
    monkeypatch.setattr(graph, "check_hallucination", lambda answer, chunks: False)


class TestAgenticRAGGraph:
    """Tests for the LangGraph agentic RAG pipeline."""

    def test_run_agentic_rag_returns_complete_state(self, stub_agents, sample_chat_history):
        """Should return complete state with all fields."""
        result = run_agentic_rag(
            "What is LangGraph?", chat_history=sample_chat_history, doc_ids=["doc_1"]
        )

        assert result["question"] == "What is LangGraph?"
        assert result["doc_ids"] == ["doc_1"]
        assert result["answer"] == "Mocked answer [1]"
        assert "agent_trace" in result
        assert "hallucination_warning" in result
        assert len(result["agent_trace"]) == 5

    def test_agent_trace_contains_all_nodes(self, stub_agents):
        """Agent trace should contain all 5 nodes in order."""
        result = run_agentic_rag("test", [], ["doc_1"])
        trace = result["agent_trace"]

        assert trace == [
            "query_rewriter",
            "retriever",
            "relevance_grader",
            "answer_generator",
            "hallucination_checker",
        ]

    def test_rewritten_query_is_used_for_retrieval(self, stub_agents, monkeypatch):
        """Retrieval should run against the rewritten query, not the raw question."""
        captured: dict[str, str] = {}

        def capture_retrieve(query, doc_ids=None, top_k=None):
            captured["query"] = query
            return [_chunk(0, "LangGraph is a library for building stateful applications.")]

        monkeypatch.setattr(graph, "retrieve_chunks", capture_retrieve)

        result = run_agentic_rag("What is LangGraph?", [], ["doc_1"])

        assert captured["query"] == "What is LangGraph? (rewritten)"
        assert result["rewritten_query"] == "What is LangGraph? (rewritten)"

    def test_irrelevant_chunks_are_filtered(self, stub_agents, monkeypatch):
        """Only chunks graded relevant should reach the answer generator."""
        captured: dict[str, list[dict[str, Any]]] = {}

        def capture_generate(question, chunks):
            captured["chunks"] = chunks
            return "Mocked answer [1]"

        monkeypatch.setattr(graph, "generate_answer", capture_generate)

        result = run_agentic_rag("What is LangGraph?", [], ["doc_1"])

        assert len(result["retrieved_chunks"]) == 2
        assert len(result["relevant_chunks"]) == 1
        assert captured["chunks"] == result["relevant_chunks"]

    def test_hallucination_warning_when_detected(self, stub_agents, monkeypatch):
        """Should include warning when hallucination detected."""
        monkeypatch.setattr(graph, "check_hallucination", lambda answer, chunks: True)

        result = run_agentic_rag("test", [], ["doc_1"])

        assert result["hallucination_warning"] is not None
        assert "WARNING" in result["hallucination_warning"]

    def test_no_warning_when_grounded(self, stub_agents):
        """Should not include warning when answer is grounded."""
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
            "hallucination_warning": None,
            "node_latencies": {},
        }
        assert "question" in state
        assert "hallucination_warning" in state
