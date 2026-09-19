"""
Query Rewriter agent — resolves conversational references to build a standalone search query.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .llm_factory import get_llm


def rewrite_query(question: str, chat_history: list[dict[str, Any]]) -> str:
    """
    Formulates a standalone search query based on the current question and history.
    If there is no chat history, returns the original question.
    """
    if not chat_history:
        return question

    llm: BaseChatModel = get_llm(temperature=0.0)

    # Format chat history as a string
    history_lines: list[str] = []
    for msg in chat_history:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        history_lines.append(f"{role}: {content}")
    history_str = "\n".join(history_lines)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert search query reformulator. "
                    "Given a conversation history and a follow-up question, your task is to "
                    "rewrite the follow-up question into a standalone question (in the same language) "
                    "that captures all relevant context from the history. "
                    "Do NOT answer the question. Just output the rewritten query."
                ),
            ),
            (
                "human",
                (
                    "Conversation History:\n"
                    "{chat_history}\n\n"
                    "Follow-up Question: {question}\n\n"
                    "Standalone Query (Only return the text of the rewritten query):"
                ),
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"chat_history": history_str, "question": question})

    return rewritten.strip()
