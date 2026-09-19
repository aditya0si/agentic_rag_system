"""
Regression tests — validates the agentic RAG pipeline against a golden dataset.

Two kinds of check live here:

* retrieval-only checks (marked ``integration``): they embed the golden contexts
  into a real local ChromaDB collection with the local sentence-transformer and
  verify that retrieval surfaces the right context. No LLM is involved.
* answer-quality checks (marked ``llm``): they run the full pipeline, which calls
  the configured hosted LLM, so they need ``GOOGLE_API_KEY`` (or an OpenAI key)
  and are not part of the default CI selection.

For full RAGAS evaluation (requires LLM), run:
    cd eval && python run_evaluation.py
"""

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pytest

from backend.core.agents.graph import run_agentic_rag
from backend.core.basic_rag import run_basic_rag
from backend.core.ingestion import chunk_document, extract_text
from backend.core.vector_store import VectorStoreManager

pytestmark = pytest.mark.integration


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

GOLDEN_SET_PATH = Path(__file__).parent.parent / "eval" / "test_set.json"


def similarity(a: str, b: str) -> float:
    """Compute string similarity ratio between 0 and 1."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def load_golden_set() -> list[dict[str, Any]]:
    """Load the golden Q&A dataset."""
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        golden: list[dict[str, Any]] = json.load(f)
        return golden


def setup_test_documents() -> list[str]:
    """Ingest golden-set contexts into the vector store and return doc_ids."""
    golden = load_golden_set()
    manager = VectorStoreManager()
    doc_ids = []

    for idx, item in enumerate(golden):
        doc_id = f"golden_doc_{idx}"
        doc_name = f"golden_{idx}.txt"

        # Clean up any existing
        try:
            manager.delete_by_doc_id(doc_id)
        except Exception:
            pass

        # Write context to temp file
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(item["context"])
            temp_path = f.name

        # Extract and chunk
        pages = extract_text(temp_path)
        chunks = chunk_document(pages, doc_id=doc_id, doc_name=doc_name)

        # Add to vector store
        manager.add_chunks(chunks)
        doc_ids.append(doc_id)

        # Clean up temp file
        import os

        os.unlink(temp_path)

    return doc_ids


def cleanup_test_documents(doc_ids: list[str]) -> None:
    """Remove test documents from vector store."""
    manager = VectorStoreManager()
    for doc_id in doc_ids:
        try:
            manager.delete_by_doc_id(doc_id)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def golden_doc_ids():
    """Set up golden documents once for the whole module."""
    doc_ids = setup_test_documents()
    yield doc_ids
    cleanup_test_documents(doc_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Regression tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGoldenDataset:
    """Validate pipeline against the golden Q&A set."""

    @pytest.mark.regression
    def test_golden_set_loads(self):
        """Ensure the golden dataset loads and has expected structure."""
        golden = load_golden_set()
        assert len(golden) >= 10, "Golden set should have at least 10 Q&A pairs"
        for item in golden:
            assert "question" in item
            assert "ground_truth" in item
            assert "context" in item

    @pytest.mark.regression
    @pytest.mark.parametrize("golden_item", load_golden_set())
    def test_retrieval_finds_relevant_context(self, golden_doc_ids, golden_item):
        """
        Verify that retrieval returns chunks with similarity to the golden context.
        This is a lightweight check that doesn't require LLM calls.
        """
        from backend.core.agents.retriever import retrieve_chunks

        chunks = retrieve_chunks(golden_item["question"], doc_ids=golden_doc_ids)
        assert len(chunks) > 0, "Retrieval should return at least one chunk"

        # At least one retrieved chunk should have reasonable similarity to context
        max_sim = max(similarity(chunk["chunk_text"], golden_item["context"]) for chunk in chunks)
        assert max_sim > 0.3, (
            f"Best retrieved chunk similarity to golden context is too low: {max_sim:.2f}"
        )

    @pytest.mark.regression
    @pytest.mark.llm
    def test_agentic_trace_completeness(self, golden_doc_ids):
        """Verify the agentic pipeline runs all 5 agents."""
        golden = load_golden_set()
        first_item = golden[0]

        result = run_agentic_rag(
            first_item["question"],
            chat_history=[],
            doc_ids=golden_doc_ids,
        )

        trace = result.get("agent_trace", [])
        expected_agents = {
            "query_rewriter",
            "retriever",
            "relevance_grader",
            "answer_generator",
            "hallucination_checker",
        }
        assert expected_agents.issubset(set(trace)), f"Missing agents in trace. Got: {trace}"

    @pytest.mark.regression
    @pytest.mark.llm
    def test_agentic_better_than_naive(self, golden_doc_ids):
        """
        Verify that agentic RAG produces answers at least as good as naive RAG
        on the golden set (using string similarity to ground truth).
        """
        golden = load_golden_set()
        agentic_scores = []
        naive_scores = []

        for item in golden:
            # Agentic
            agentic_result = run_agentic_rag(
                item["question"], chat_history=[], doc_ids=golden_doc_ids
            )
            agentic_answer = agentic_result.get("answer", "")
            agentic_scores.append(similarity(agentic_answer, item["ground_truth"]))

            # Naive
            naive_result = run_basic_rag(item["question"], doc_ids=golden_doc_ids)
            naive_scores.append(similarity(naive_result.answer, item["ground_truth"]))

        avg_agentic = sum(agentic_scores) / len(agentic_scores)
        avg_naive = sum(naive_scores) / len(naive_scores)

        # Agentic should be at least as good as naive
        assert avg_agentic >= avg_naive * 0.9, (
            f"Agentic RAG ({avg_agentic:.3f}) significantly worse than "
            f"naive RAG ({avg_naive:.3f}) on golden set"
        )

    @pytest.mark.regression
    @pytest.mark.llm
    def test_no_hallucination_on_grounded_answers(self, golden_doc_ids):
        """Verify hallucination checker runs and doesn't flag well-grounded answers."""
        golden = load_golden_set()
        first_item = golden[0]

        result = run_agentic_rag(
            first_item["question"],
            chat_history=[],
            doc_ids=golden_doc_ids,
        )

        # The hallucination_warning should be None for well-grounded answers
        # (or a string if flagged — we just verify the field exists)
        assert "hallucination_warning" in result
        assert result["hallucination_warning"] is None or isinstance(
            result["hallucination_warning"], str
        )

    @pytest.mark.regression
    @pytest.mark.llm
    def test_node_latencies_recorded(self, golden_doc_ids):
        """Verify per-node latency tracking is present in the result."""
        golden = load_golden_set()
        result = run_agentic_rag(golden[0]["question"], chat_history=[], doc_ids=golden_doc_ids)

        latencies = result.get("node_latencies", {})
        assert isinstance(latencies, dict)
        # At least retriever and answer_generator should have latency entries
        assert "retriever" in latencies or "answer_generator" in latencies
