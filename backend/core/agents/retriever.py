"""
Retrieval agent — wraps ChromaDB similarity search, returns top-k chunks.
"""

from typing import Any

from ..vector_store import VectorStoreManager


def retrieve_chunks(
    query: str, doc_ids: list[str] | None = None, top_k: int | None = None
) -> list[dict[str, Any]]:
    """
    Retrieves relevant document chunks from the vector store.
    Filters by doc_ids if provided.
    """
    manager = VectorStoreManager()
    return manager.query(query, doc_ids=doc_ids, top_k=top_k)
