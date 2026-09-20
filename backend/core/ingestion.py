"""
Document ingestion pipeline — text extraction, chunking, and metadata mapping.
"""

from pathlib import Path
from typing import TypedDict

import docx
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ExtractedPage(TypedDict):
    """Represents a page of extracted text with its page number."""

    text: str
    page_number: int


class DocumentChunk(TypedDict):
    """Represents a chunk of a document with metadata."""

    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: int
    chunk_text: str
    chunk_index: int
    char_count: int


def extract_text_from_pdf(file_path: Path | str) -> list[ExtractedPage]:
    """
    Extracts text from a PDF file page by page.
    Returns a list of dicts: [{'text': str, 'page_number': int}]
    """
    pages: list[ExtractedPage] = []
    with pdfplumber.open(file_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"text": text, "page_number": idx + 1})
    return pages


def extract_text_from_docx(file_path: Path | str) -> list[ExtractedPage]:
    """
    Extracts text from a DOCX file.
    DOCX has no page structure, so it returns a single page.
    """
    doc = docx.Document(str(file_path))
    full_text = "\n".join([para.text for para in doc.paragraphs])
    return [{"text": full_text, "page_number": 1}]


def extract_text_from_txt(file_path: Path | str) -> list[ExtractedPage]:
    """
    Extracts text from a plain TXT file.
    TXT has no page structure, so it returns a single page.
    """
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{"text": text, "page_number": 1}]


def extract_text(file_path: Path | str) -> list[ExtractedPage]:
    """
    Dispatcher to extract text based on file extension.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    elif suffix == ".docx":
        return extract_text_from_docx(path)
    elif suffix in (".txt", ".md"):
        return extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")


def chunk_document(
    pages: list[ExtractedPage],
    doc_id: str,
    doc_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    """
    Chunks document pages into chunks of chunk_size characters with chunk_overlap overlap.
    Maintains page association.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: list[DocumentChunk] = []
    chunk_index = 0
    for page in pages:
        text = page["text"]
        page_num = page["page_number"]

        # Split text of this page
        page_chunks = splitter.split_text(text)

        for chunk_text in page_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunk_id = f"{doc_id}_chunk_{chunk_index:03d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page_number": page_num,
                    "chunk_text": chunk_text,
                    "chunk_index": chunk_index,
                    "char_count": len(chunk_text),
                }
            )
            chunk_index += 1

    return chunks
