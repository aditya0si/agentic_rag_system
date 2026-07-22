"""
Agentic RAG Research Assistant - Backend Core Package.

Exports:
- ingestion: Document text extraction and chunking
- vector_store: ChromaDB vector store management
- agents: LangGraph agent implementations
- memory: Session and chat history management
"""

from . import ingestion
from . import vector_store
from . import agents
from . import memory

__all__ = [
    "ingestion",
    "vector_store", 
    "agents",
    "memory",
]