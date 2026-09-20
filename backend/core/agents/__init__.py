"""
Agents package — LangGraph agent nodes for the RAG pipeline.

Exports:
- query_rewriter: Reformulates conversational queries
- retriever: ChromaDB similarity search wrapper
- relevance_grader: Filters irrelevant retrieved chunks
- answer_generator: Produces grounded, cited answers
- hallucination_checker: Validates answer groundedness
- graph: Compiled LangGraph StateGraph pipeline
- llm_factory: LLM factory module with cached get_llm
"""

from .answer_generator import generate_answer
from .graph import AgentState, run_agentic_rag
from .hallucination_checker import check_hallucination
from .llm_factory import get_llm
from .query_rewriter import rewrite_query
from .relevance_grader import grade_chunk_relevance
from .retriever import retrieve_chunks

__all__ = [
    "rewrite_query",
    "retrieve_chunks",
    "grade_chunk_relevance",
    "generate_answer",
    "check_hallucination",
    "run_agentic_rag",
    "AgentState",
    "get_llm",
]
