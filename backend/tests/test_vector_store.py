"""
Integration tests for Vector Store operations.

These exercise the real embedding model and a real (local, persistent) ChromaDB
collection, hence the ``integration`` marker: they download the
``all-MiniLM-L6-v2`` sentence-transformer on first run and are much slower than
the unit suite.
"""

import pytest

from backend.core.vector_store import VectorStoreManager

pytestmark = pytest.mark.integration


class TestVectorStoreManager:
    """Integration tests for VectorStoreManager."""

    def test_add_and_query_chunks(self, vector_store_manager, sample_chunks):
        """Should add chunks and retrieve them via query."""
        vector_store_manager.add_chunks(sample_chunks)

        results = vector_store_manager.query("What is LangGraph?", top_k=2)

        assert len(results) > 0
        assert results[0]["doc_id"] == "doc_test"
        assert "chunk_text" in results[0]
        assert "score" in results[0]

    def test_query_filters_by_doc_id(self, vector_store_manager, sample_chunks):
        """Should filter results by doc_ids."""
        vector_store_manager.add_chunks(sample_chunks)

        results = vector_store_manager.query("LangGraph", doc_ids=["doc_test"], top_k=5)

        assert all(r["doc_id"] == "doc_test" for r in results)

    def test_delete_by_doc_id(self, vector_store_manager, sample_chunks):
        """Should delete all chunks for a document."""
        vector_store_manager.add_chunks(sample_chunks)

        results_before = vector_store_manager.query("LangGraph", doc_ids=["doc_test"])
        assert len(results_before) > 0

        vector_store_manager.delete_by_doc_id("doc_test")

        results_after = vector_store_manager.query("LangGraph", doc_ids=["doc_test"])
        assert len(results_after) == 0

    def test_empty_query_returns_empty_list(self, monkeypatch):
        """Query on an empty collection should return an empty list.

        Uses a dedicated collection so the assertion does not depend on what
        other tests have written to the shared collection in this session.
        """
        manager = VectorStoreManager()
        monkeypatch.setattr(manager, "collection_name", "research_assistant_empty_check")

        results = manager.query("nonexistent query")

        assert results == []

    def test_get_vectorstore_returns_chroma_instance(self, vector_store_manager):
        """get_vectorstore should return a Chroma instance."""
        from langchain_community.vectorstores import Chroma

        vs = vector_store_manager.get_vectorstore()
        assert isinstance(vs, Chroma)
