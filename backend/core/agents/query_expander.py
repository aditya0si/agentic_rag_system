"""
Query Expansion agent — generates multiple query variations to improve recall.

Uses LLM to generate semantic variations of the user query for better retrieval.
"""

from typing import Any
from .llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import structlog

logger = structlog.get_logger(__name__)


def expand_query(question: str, num_variations: int = 3) -> list[str]:
    """
    Generate semantic variations of the query to improve retrieval recall.
    
    Args:
        question: Original user question
        num_variations: Number of query variations to generate
        
    Returns:
        List of query variations (including original)
    """
    logger.info("expanding_query", question=question[:100])
    
    llm = get_llm(temperature=0.7)  # Higher temperature for diversity
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert at generating search query variations. "
            "Given a question, generate {num_variations} alternative phrasings "
            "that preserve the meaning but use different words and structures. "
            "Return ONLY the variations, one per line, without numbering or explanations."
        )),
        ("human", "Question: {question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({
            "question": question,
            "num_variations": num_variations
        })
        
        # Split by newlines and clean
        variations = [line.strip() for line in result.split("\n") if line.strip()]
        
        # Always include original question
        all_queries = [question] + variations
        
        logger.info(
            "query_expanded",
            original=question[:50],
            num_variations=len(variations)
        )
        
        return all_queries
        
    except Exception as e:
        logger.warning("query_expansion_failed", error=str(e))
        # Fallback: return original question only
        return [question]
