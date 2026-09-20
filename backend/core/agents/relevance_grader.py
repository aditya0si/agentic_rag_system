"""
Relevance Grader agent — scores document chunks as relevant/irrelevant to filter out noise.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .llm_factory import get_llm


class GradeChunk(BaseModel):
    """Binary score for relevance of a document chunk."""

    binary_score: str = Field(
        ..., description="Relevance score: 'yes' (relevant) or 'no' (not relevant)"
    )


def _binary_score(result: "GradeChunk | dict[str, Any] | BaseModel") -> str:
    """Read the binary score out of a structured-output result.

    ``with_structured_output`` returns either the pydantic model or a plain dict
    depending on the provider integration, so both shapes are accepted.
    """
    if isinstance(result, GradeChunk):
        return result.binary_score
    if isinstance(result, dict):
        return str(result.get("binary_score", ""))
    # Any other model returned by a provider integration: read it defensively.
    return str(getattr(result, "binary_score", ""))


def grade_chunk_relevance(query: str, chunk_text: str) -> bool:
    """
    Grades the relevance of a retrieved chunk to the query.
    Returns True if relevant ('yes'), False otherwise ('no').
    """
    llm: BaseChatModel = get_llm(temperature=0.0)

    try:
        structured_llm = llm.with_structured_output(GradeChunk)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an objective grader assessing whether a retrieved document chunk is relevant "
                        "to a search query. Evaluate if the chunk contains context, details, or direct information "
                        "that could assist in answering the query.\n\n"
                        "Provide a binary score: 'yes' if relevant, or 'no' if irrelevant."
                    ),
                ),
                ("human", "Search Query: {query}\n\nDocument Chunk:\n{chunk}"),
            ]
        )

        chain = prompt | structured_llm
        result = chain.invoke({"query": query, "chunk": chunk_text})
        return _binary_score(result).strip().lower() == "yes"

    except Exception:
        # Fallback to plain text evaluation if structured output fails/unsupported
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an objective grader assessing whether a retrieved document chunk is relevant "
                        "to a search query. Evaluate if the chunk contains context, details, or direct information "
                        "that could assist in answering the query.\n\n"
                        "Respond with exactly one word: 'yes' if it is relevant, or 'no' if it is not."
                    ),
                ),
                ("human", "Search Query: {query}\n\nDocument Chunk:\n{chunk}"),
            ]
        )
        chain = prompt | llm
        result = chain.invoke({"query": query, "chunk": chunk_text})
        # Handle string response (from base LLM wrapper)
        content = getattr(result, "content", str(result)).strip().lower()
        return "yes" in content
