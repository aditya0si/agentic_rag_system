"""
LLM Factory — centralized LLM client creation with caching.
"""

from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ...config import (
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0) -> BaseChatModel:
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
            api_key=SecretStr(OPENAI_API_KEY),
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
