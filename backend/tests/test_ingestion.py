"""
Unit tests for Document Ingestion pipeline.

Everything here runs against local temporary files (txt/docx/pdf parsing and
chunking); no external service or model download is involved.
"""

import pytest

from backend.core.ingestion import (
    ExtractedPage,
    chunk_document,
    extract_text,
    extract_text_from_docx,
    extract_text_from_txt,
)

pytestmark = pytest.mark.unit


class TestDocumentIngestion:
    """Tests for document ingestion pipeline."""

    def test_extract_text_from_txt(self, tmp_path):
        """Should extract text from TXT file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is a test document.\nWith multiple lines.")

        pages = extract_text_from_txt(txt_file)

        assert len(pages) == 1
        assert pages[0]["page_number"] == 1
        assert "This is a test document" in pages[0]["text"]

    def test_extract_text_from_docx(self, tmp_path):
        """Should extract text from DOCX file."""
        import docx

        docx_file = tmp_path / "test.docx"
        doc = docx.Document()
        doc.add_paragraph("This is a test paragraph.")
        doc.add_paragraph("Another paragraph here.")
        doc.save(docx_file)

        pages = extract_text_from_docx(docx_file)

        assert len(pages) == 1
        assert pages[0]["page_number"] == 1
        assert "This is a test paragraph" in pages[0]["text"]

    def test_extract_text_dispatcher(self, tmp_path):
        """extract_text dispatcher should route to correct extractor."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Test content")

        pages = extract_text(txt_file)

        assert len(pages) == 1
        assert "Test content" in pages[0]["text"]

    def test_extract_text_unsupported_extension(self, tmp_path):
        """Should raise ValueError for unsupported extensions."""
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("content")

        with pytest.raises(ValueError, match="Unsupported file extension"):
            extract_text(unsupported_file)

    def test_chunk_document(self):
        """Should chunk document pages into chunks with metadata."""
        pages: list[ExtractedPage] = [
            {"text": "This is a test document. " * 20, "page_number": 1},
            {"text": "Second page content. " * 15, "page_number": 2},
        ]

        chunks = chunk_document(
            pages, doc_id="doc_test", doc_name="test.txt", chunk_size=100, chunk_overlap=20
        )

        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "doc_id" in chunk
            assert "doc_name" in chunk
            assert "page_number" in chunk
            assert "chunk_text" in chunk
            assert "chunk_index" in chunk
            assert "char_count" in chunk
            assert chunk["doc_id"] == "doc_test"
            assert chunk["doc_name"] == "test.txt"
            assert chunk["chunk_index"] >= 0

    def test_chunk_document_empty_pages(self):
        """Should handle empty pages gracefully."""
        pages: list[ExtractedPage] = [{"text": "", "page_number": 1}]

        chunks = chunk_document(pages, doc_id="doc_test", doc_name="test.txt")

        assert chunks == []

    def test_chunk_document_preserves_page_numbers(self):
        """Chunks should preserve correct page numbers."""
        pages: list[ExtractedPage] = [
            {"text": "Page 1 content. " * 20, "page_number": 1},
            {"text": "Page 2 content. " * 20, "page_number": 2},
        ]

        chunks = chunk_document(pages, doc_id="doc_test", doc_name="test.txt", chunk_size=100)

        page_numbers = {c["page_number"] for c in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers
