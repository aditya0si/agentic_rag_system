"""
Basic RAG Chain (naive RAG baseline).
Wires retriever directly to generator without query rewriter or relevance grader agents.
"""

from ..models.schemas import AskResponse, Citation
from .agents.answer_generator import generate_answer
from .agents.retriever import retrieve_chunks


def run_basic_rag(question: str, doc_ids: list[str] | None = None) -> AskResponse:
    """
    Executes a simple naive retrieve-then-generate pipeline.
    """
    # 1. Retrieve chunks
    chunks = retrieve_chunks(question, doc_ids=doc_ids)

    # 2. Generate answer
    answer = generate_answer(question, chunks)

    # 3. Build citations
    citations = []
    if chunks and "I couldn't find relevant information" not in answer:
        for c in chunks:
            citations.append(
                Citation(
                    doc_id=c["doc_id"],
                    doc_name=c.get("doc_name", ""),
                    page=c.get("page_number", 1),
                    chunk_text=c["chunk_text"],
                )
            )

    return AskResponse(
        answer=answer, citations=citations, agent_trace=["retriever", "answer_generator"]
    )
