"""
LLM Factory — centralized LLM client creation with caching.
"""

from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from ...config import (
    LLM_PROVIDER,
    OPENAI_MODEL,
    OPENAI_API_KEY,
    GOOGLE_MODEL,
    GOOGLE_API_KEY,
)


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0):
    """
    Returns the configured LLM client with caching.
    
    Args:
        temperature: Sampling temperature (0.0 = deterministic)
        
    Returns:
        Configured LLM client instance
    """
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=temperature,
            api_key=OPENAI_API_KEY,
        )
    elif LLM_PROVIDER == "google":
        return ChatGoogleGenerativeAI(
            model=GOOGLE_MODEL,
            temperature=temperature,
            api_key=GOOGLE_API_KEY,
            max_retries=15,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")