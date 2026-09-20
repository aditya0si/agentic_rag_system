"""
Pytest configuration and shared fixtures.

Provides common fixtures for all tests. Markers are declared in the
repository-root ``pytest.ini`` (single source of truth) and enforced by the
root ``conftest.py``.
"""

from pathlib import Path

import pytest

from backend.core.ingestion import chunk_document
from backend.core.vector_store import VectorStoreManager


@pytest.fixture(scope="session")
def vector_store_manager():
    """Create a VectorStoreManager instance for testing."""
    manager = VectorStoreManager()
    yield manager
    # Cleanup: delete test collections
    try:
        manager.delete_by_doc_id("doc_test")
        manager.delete_by_doc_id("doc_hal_test")
        manager.delete_by_doc_id("doc_agentic_test")
        manager.delete_by_doc_id("doc_rag_test")
        manager.delete_by_doc_id("doc_txt_01")
        manager.delete_by_doc_id("doc_docx_01")
        manager.delete_by_doc_id("doc_pdf_01")
    except Exception:
        pass


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing."""
    return [
        {
            "chunk_id": "doc_test_chunk_000",
            "doc_id": "doc_test",
            "doc_name": "test.txt",
            "page_number": 1,
            "chunk_text": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
            "chunk_index": 0,
            "char_count": 76,
        },
        {
            "chunk_id": "doc_test_chunk_001",
            "doc_id": "doc_test",
            "doc_name": "test.txt",
            "page_number": 1,
            "chunk_text": "It extends the LangChain Expression Language (LCEL) with multi-agent coordination.",
            "chunk_index": 1,
            "char_count": 80,
        },
    ]


@pytest.fixture
def temp_test_file(tmp_path):
    """Create a temporary test text file."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("This is a test document for testing the ingestion pipeline.")
    return file_path


@pytest.fixture
def sample_pages():
    """Sample extracted pages for chunking tests."""
    return [
        {
            "text": "LangGraph is a library for building stateful applications. " * 10,
            "page_number": 1,
        },
        {"text": "ChromaDB is a vector database for semantic search. " * 10, "page_number": 2},
    ]


@pytest.fixture
def sample_chunks_from_pages(sample_pages):
    """Create chunks from sample pages."""
    return chunk_document(sample_pages, doc_id="doc_test", doc_name="test.txt")


@pytest.fixture
def sample_chat_history():
    """Two-turn chat history for agent tests."""
    return [
        {"role": "user", "content": "Tell me about LangGraph"},
        {"role": "assistant", "content": "LangGraph is a library for building stateful apps"},
    ]


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Small text file used by ingestion tests."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a test document.\nWith multiple lines.")
    return file_path
