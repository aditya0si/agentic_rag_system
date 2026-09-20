"""
LangGraph StateGraph definition — manages states, transitions, and agent nodes.
"""

from typing import Any, TypedDict

import structlog
from langgraph.graph import END, START, StateGraph

from ..metrics import metrics
from .answer_generator import generate_answer
from .hallucination_checker import check_hallucination
from .query_rewriter import rewrite_query
from .relevance_grader import grade_chunk_relevance
from .reranker import rerank_chunks
from .retriever import retrieve_chunks

logger = structlog.get_logger(__name__)


class AgentState(TypedDict):
    """
    Represents the state of the agentic RAG pipeline.
    """

    question: str
    doc_ids: list[str] | None
    chat_history: list[dict[str, str]]
    rewritten_query: str
    retrieved_chunks: list[dict[str, Any]]
    relevant_chunks: list[dict[str, Any]]
    answer: str
    agent_trace: list[str]
    hallucination_warning: str | None
    node_latencies: dict[str, float]  # Per-node latency for observability


# =============================================================================
# Node Functions
# =============================================================================


def rewrite_node(state: AgentState) -> dict[str, Any]:
    """
    Query Rewriting node — reformulates the user question based on chat history.
    """
    question = state["question"]
    chat_history = state.get("chat_history", [])

    rewritten = rewrite_query(question, chat_history)

    trace = list(state.get("agent_trace", []))
    trace.append("query_rewriter")

    return {"rewritten_query": rewritten, "agent_trace": trace}


def retrieve_node(state: AgentState) -> dict[str, Any]:
    """
    Retrieval node — fetches top-k document chunks matching the rewritten query,
    then re-ranks them using a cross-encoder for improved precision.
    """
    import time

    start = time.time()

    query = state["rewritten_query"]
    doc_ids = state.get("doc_ids")

    # Over-fetch 2x then re-rank down to top-k for better precision
    chunks = retrieve_chunks(query, doc_ids=doc_ids)
    chunks = rerank_chunks(query, chunks)

    latency = time.time() - start
    metrics.record_histogram("retrieval_duration_seconds", latency)
    metrics.increment_counter("retrieval_chunks_total", len(chunks))
    logger.info("retrieval_complete", num_chunks=len(chunks), latency_s=round(latency, 3))

    trace = list(state.get("agent_trace", []))
    trace.append("retriever")

    latencies = dict(state.get("node_latencies", {}))
    latencies["retriever"] = round(latency, 3)

    return {
        "retrieved_chunks": chunks,
        "agent_trace": trace,
        "node_latencies": latencies,
    }


def grade_node(state: AgentState) -> dict[str, Any]:
    """
    Relevance Grading node — evaluates retrieved chunks and filters out irrelevant ones.
    """
    import time

    start = time.time()

    query = state["rewritten_query"]
    chunks = state.get("retrieved_chunks", [])

    relevant_chunks = []
    for c in chunks:
        is_relevant = grade_chunk_relevance(query, c["chunk_text"])
        if is_relevant:
            relevant_chunks.append(c)
        time.sleep(2)  # Spacing out requests to avoid 429 Resource Exhausted on free tier

    latency = time.time() - start
    metrics.record_histogram("grading_duration_seconds", latency)
    metrics.increment_counter("grading_filtered_total", len(chunks) - len(relevant_chunks))
    logger.info(
        "grading_complete",
        input_chunks=len(chunks),
        relevant_chunks=len(relevant_chunks),
        latency_s=round(latency, 3),
    )

    trace = list(state.get("agent_trace", []))
    trace.append("relevance_grader")

    latencies = dict(state.get("node_latencies", {}))
    latencies["relevance_grader"] = round(latency, 3)

    return {
        "relevant_chunks": relevant_chunks,
        "agent_trace": trace,
        "node_latencies": latencies,
    }


def generate_node(state: AgentState) -> dict[str, Any]:
    """
    Answer Generation node — produces a grounded response from relevant chunks.
    """
    import time

    start = time.time()

    query = state["rewritten_query"]
    chunks = state.get("relevant_chunks", [])

    answer = generate_answer(query, chunks)

    latency = time.time() - start
    metrics.record_histogram("generation_duration_seconds", latency)
    logger.info("generation_complete", answer_length=len(answer), latency_s=round(latency, 3))

    trace = list(state.get("agent_trace", []))
    trace.append("answer_generator")

    latencies = dict(state.get("node_latencies", {}))
    latencies["answer_generator"] = round(latency, 3)

    return {
        "answer": answer,
        "agent_trace": trace,
        "node_latencies": latencies,
    }


def hallucination_node(state: AgentState) -> dict[str, Any]:
    """
    Hallucination Checker node — cross-checks the generated answer against source contexts.
    """
    answer = state.get("answer", "")
    chunks = state.get("relevant_chunks", [])

    # Run the hallucination check
    is_hallucinated = check_hallucination(answer, chunks)

    warning = None
    if is_hallucinated:
        warning = "WARNING: Some claims may not be fully supported by the source documents"

    trace = list(state.get("agent_trace", []))
    trace.append("hallucination_checker")

    return {"hallucination_warning": warning, "agent_trace": trace}


# =============================================================================
# Graph Construction
# =============================================================================

workflow = StateGraph(AgentState)

# Register nodes
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("generate", generate_node)
workflow.add_node("hallucination_check", hallucination_node)

# Connect edges
workflow.add_edge(START, "rewrite")
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_edge("grade", "generate")
workflow.add_edge("generate", "hallucination_check")
workflow.add_edge("hallucination_check", END)

# Compile the pipeline
agentic_pipeline = workflow.compile()

# =============================================================================
# Execution Helper
# =============================================================================


def run_agentic_rag(
    question: str, chat_history: list[dict[str, str]], doc_ids: list[str] | None = None
) -> dict[str, Any]:
    """
    Invokes the Compiled LangGraph agentic RAG pipeline.
    """
    initial_state: AgentState = {
        "question": question,
        "doc_ids": doc_ids,
        "chat_history": chat_history,
        "rewritten_query": "",
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "answer": "",
        "agent_trace": [],
        "hallucination_warning": None,
        "node_latencies": {},
    }
    result: dict[str, Any] = agentic_pipeline.invoke(initial_state)
    return result
