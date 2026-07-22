"""
Integration tests for Vector Store operations.
"""

import pytest
from core.vector_store import VectorStoreManager


class TestVectorStoreManager:
    """Integration tests for VectorStoreManager."""

    def test_add_and_query_chunks(self, vector_store_manager, sample_chunks):
        """Should add chunks and retrieve them via query."""
        # Add chunks
        vector_store_manager.add_chunks(sample_chunks)
        
        # Query for relevant chunks
        results = vector_store_manager.query("What is LangGraph?", top_k=2)
        
        assert len(results) > 0
        assert results[0]["doc_id"] == "doc_test"
        assert "chunk_text" in results[0]
        assert "score" in results[0]

    def test_query_filters_by_doc_id(self, vector_store_manager, sample_chunks):
        """Should filter results by doc_ids."""
        vector_store_manager.add_chunks(sample_chunks)
        
        results = vector_store_manager.query(
            "LangGraph", 
            doc_ids=["doc_test"], 
            top_k=5
        )
        
        assert all(r["doc_id"] == "doc_test" for r in results)

    def test_delete_by_doc_id(self, vector_store_manager, sample_chunks):
        """Should delete all chunks for a document."""
        vector_store_manager.add_chunks(sample_chunks)
        
        # Verify chunks exist
        results_before = vector_store_manager.query("LangGraph", doc_ids=["doc_test"])
        assert len(results_before) > 0
        
        # Delete
        vector_store_manager.delete_by_doc_id("doc_test")
        
        # Verify chunks are gone
        results_after = vector_store_manager.query("LangGraph", doc_ids=["doc_test"])
        assert len(results_after) == 0

    def test_empty_query_returns_empty_list(self, vector_store_manager):
        """Query on empty collection should return empty list."""
        results = vector_store_manager.query("nonexistent query")
        assert results == []

    def test_get_vectorstore_returns_chroma_instance(self, vector_store_manager):
        """get_vectorstore should return a Chroma instance."""
        from langchain_community.vectorstores import Chroma
        vs = vector_store_manager.get_vectorstore()
        assert isinstance(vs, Chroma)