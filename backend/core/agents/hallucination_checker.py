"""
Hallucination Checker agent — cross-checks the generated answer against source contexts.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .llm_factory import get_llm


class GradeHallucination(BaseModel):
    """Binary score for hallucination check."""

    binary_score: str = Field(
        ...,
        description="Groundedness score: 'yes' if there are unsupported claims/hallucinations, 'no' if all claims are fully supported.",
    )


def _binary_score(result: "GradeHallucination | dict[str, Any] | BaseModel") -> str:
    """Read the binary score out of a structured-output result.

    ``with_structured_output`` returns either the pydantic model or a plain dict
    depending on the provider integration, so both shapes are accepted.
    """
    if isinstance(result, GradeHallucination):
        return result.binary_score
    if isinstance(result, dict):
        return str(result.get("binary_score", ""))
    # Any other model returned by a provider integration: read it defensively.
    return str(getattr(result, "binary_score", ""))


def check_hallucination(answer: str, chunks: list[dict[str, Any]]) -> bool:
    """
    Checks if the generated answer contains unsupported claims/hallucinations.
    Returns True if hallucination detected ('yes'), False otherwise ('no').
    """
    if not chunks:
        # Default fallback "not found" is not a hallucination
        if "I couldn't find relevant information" in answer:
            return False
        return True

    llm: BaseChatModel = get_llm(temperature=0.0)

    # Format all relevant chunks
    context_str = "\n\n".join([f"Context Chunk:\n{c.get('chunk_text', '')}" for c in chunks])

    try:
        structured_llm = llm.with_structured_output(GradeHallucination)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an objective auditor assessing whether a generated answer is fully grounded "
                        "in the provided contexts. Evaluate if the answer contains any claims, facts, or "
                        "assumptions that are NOT supported by the contexts.\n\n"
                        "Provide a binary score:\n"
                        "- 'yes' if the answer contains unsupported claims, extrapolations, or hallucinations.\n"
                        "- 'no' if the answer is completely grounded in and supported by the contexts."
                    ),
                ),
                ("human", "Grounded Contexts:\n{context}\n\nGenerated Answer:\n{answer}"),
            ]
        )

        chain = prompt | structured_llm
        result = chain.invoke({"context": context_str, "answer": answer})
        return _binary_score(result).strip().lower() == "yes"

    except Exception:
        # Fallback to plain text evaluation if structured output fails
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an objective auditor assessing whether a generated answer is fully grounded "
                        "in the provided contexts. Evaluate if the answer contains any claims, facts, or "
                        "assumptions that are NOT supported by the contexts.\n\n"
                        "Respond with exactly one word:\n"
                        "- 'yes' if the answer contains unsupported claims or hallucinations.\n"
                        "- 'no' if the answer is completely grounded."
                    ),
                ),
                ("human", "Grounded Contexts:\n{context}\n\nGenerated Answer:\n{answer}"),
            ]
        )
        chain = prompt | llm
        result = chain.invoke({"context": context_str, "answer": answer})
        content = getattr(result, "content", str(result)).strip().lower()
        return "yes" in content
