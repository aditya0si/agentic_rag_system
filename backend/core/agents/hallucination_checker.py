"""
Hallucination Checker agent — cross-checks the generated answer against source contexts.
"""

from typing import Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from .llm_factory import get_llm
from langchain_core.language_models import BaseChatModel


class GradeHallucination(BaseModel):
    """Binary score for hallucination check."""
    binary_score: str = Field(
        ...,
        description="Groundedness score: 'yes' if there are unsupported claims/hallucinations, 'no' if all claims are fully supported."
    )


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
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an objective auditor assessing whether a generated answer is fully grounded "
                "in the provided contexts. Evaluate if the answer contains any claims, facts, or "
                "assumptions that are NOT supported by the contexts.\n\n"
                "Provide a binary score:\n"
                "- 'yes' if the answer contains unsupported claims, extrapolations, or hallucinations.\n"
                "- 'no' if the answer is completely grounded in and supported by the contexts."
            )),
            ("human", "Grounded Contexts:\n{context}\n\nGenerated Answer:\n{answer}")
        ])
        
        chain = prompt | structured_llm
        result = chain.invoke({"context": context_str, "answer": answer})
        return result.binary_score.strip().lower() == "yes"
        
    except Exception:
        # Fallback to plain text evaluation if structured output fails
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an objective auditor assessing whether a generated answer is fully grounded "
                "in the provided contexts. Evaluate if the answer contains any claims, facts, or "
                "assumptions that are NOT supported by the contexts.\n\n"
                "Respond with exactly one word:\n"
                "- 'yes' if the answer contains unsupported claims or hallucinations.\n"
                "- 'no' if the answer is completely grounded."
            )),
            ("human", "Grounded Contexts:\n{context}\n\nGenerated Answer:\n{answer}")
        ])
        chain = prompt | llm
        result = chain.invoke({"context": context_str, "answer": answer})
        content = getattr(result, "content", str(result)).strip().lower()
        return "yes" in content
