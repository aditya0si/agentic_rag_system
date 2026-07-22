"""
Re-ranking agent — re-orders retrieved chunks using cross-encoder for better relevance.

Uses a lightweight cross-encoder model to score query-chunk pairs more accurately
than bi-encoder similarity alone.
"""

from typing import Any
from sentence_transformers import CrossEncoder
from functools import lru_cache
import structlog

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """
    Load cross-encoder model for re-ranking.
    
    Uses a small, fast model suitable for CPU inference.
    """
    logger.info("loading_reranker_model")
    # Small, fast cross-encoder (free, local)
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int | None = None
) -> list[dict[str, Any]]:
    """
    Re-rank retrieved chunks using cross-encoder for better relevance.
    
    Args:
        query: Search query
        chunks: Retrieved chunks from vector store
        top_k: Number of top chunks to return (None = return all, re-ranked)
        
    Returns:
        Re-ranked chunks with updated scores
    """
    if not chunks:
        return chunks
    
    logger.info("reranking_chunks", num_chunks=len(chunks))
    
    # Load reranker model
    reranker = get_reranker()
    
    # Prepare query-chunk pairs
    pairs = [[query, chunk["chunk_text"]] for chunk in chunks]
    
    # Score pairs
    scores = reranker.predict(pairs)
    
    # Add rerank scores to chunks
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    
    # Sort by rerank score (descending)
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    # Return top-k if specified
    if top_k is not None:
        reranked = reranked[:top_k]
    
    logger.info(
        "reranking_complete",
        num_chunks=len(reranked),
        top_score=reranked[0]["rerank_score"] if reranked else 0
    )
    
    return reranked
