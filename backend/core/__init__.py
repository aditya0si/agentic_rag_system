"""
Agentic RAG Research Assistant - Backend Core Package.

Exports:
- ingestion: Document text extraction and chunking
- vector_store: ChromaDB vector store management
- agents: LangGraph agent implementations
- memory: Session and chat history management
"""

from . import agents, ingestion, memory, vector_store

__all__ = [
    "ingestion",
    "vector_store",
    "agents",
    "memory",
]
