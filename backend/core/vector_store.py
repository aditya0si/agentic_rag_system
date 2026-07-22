"""
Vector store interface — manages document embeddings and storage in ChromaDB.
"""

from pathlib import Path
from typing import Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ..config import (
    EMBEDDING_PROVIDER,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_API_KEY,
    HUGGINGFACE_EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
    CHROMA_PERSIST_DIR,
)
from functools import lru_cache


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """
    Initializes and returns the appropriate embedding model.
    """
    if EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
    elif EMBEDDING_PROVIDER == "huggingface":
        # HuggingFace uses all-MiniLM-L6-v2 locally (free)
        return HuggingFaceEmbeddings(
            model_name=HUGGINGFACE_EMBEDDING_MODEL
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")


class VectorStoreManager:
    """Manages the connection, insertion, and querying of ChromaDB."""
    
    def __init__(self) -> None:
        self.embeddings: Embeddings = get_embeddings()
        self.persist_directory: str = str(Path(CHROMA_PERSIST_DIR).resolve())
        self.collection_name: str = "research_assistant"
        
    def get_vectorstore(self) -> Chroma:
        """
        Returns the Chroma vector store instance.
        """
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """
        Adds document chunks to the Chroma vector store.
        """
        vectorstore = self.get_vectorstore()
        documents: list[Document] = []
        for c in chunks:
            doc = Document(
                page_content=c["chunk_text"],
                metadata={
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "doc_name": c["doc_name"],
                    "page_number": c["page_number"],
                    "chunk_index": c["chunk_index"],
                    "char_count": c["char_count"]
                }
            )
            documents.append(doc)
            
        vectorstore.add_documents(documents)
        
    def query(
        self, 
        query_text: str, 
        doc_ids: list[str] | None = None, 
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Queries Chroma for similar chunks.
        Filters by doc_ids if provided.
        """
        vectorstore = self.get_vectorstore()
        if top_k is None:
            top_k = RETRIEVAL_TOP_K
            
        # Build filter if doc_ids are provided
        search_filter: dict[str, Any] | None = None
        if doc_ids:
            if len(doc_ids) == 1:
                search_filter = {"doc_id": doc_ids[0]}
            else:
                # Chroma syntax for OR queries uses "$or"
                search_filter = {"$or": [{"doc_id": did} for did in doc_ids]}
                
        results = vectorstore.similarity_search_with_score(
            query=query_text,
            k=top_k,
            filter=search_filter
        )
        
        # Format results as simple dicts
        chunks: list[dict[str, Any]] = []
        for doc, score in results:
            chunks.append({
                "chunk_id": doc.metadata.get("chunk_id"),
                "doc_id": doc.metadata.get("doc_id"),
                "doc_name": doc.metadata.get("doc_name"),
                "page_number": doc.metadata.get("page_number", 1),
                "chunk_text": doc.page_content,
                "chunk_index": doc.metadata.get("chunk_index"),
                "char_count": doc.metadata.get("char_count"),
                "score": float(score)
            })
        return chunks

    def delete_by_doc_id(self, doc_id: str) -> None:
        """
        Deletes all chunks belonging to a specific doc_id.
        """
        vectorstore = self.get_vectorstore()
        collection = vectorstore._collection
        collection.delete(where={"doc_id": doc_id})
