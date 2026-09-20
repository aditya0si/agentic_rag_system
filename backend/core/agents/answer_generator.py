"""
Answer Generator agent — produces grounded, source-cited responses.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .llm_factory import get_llm


def generate_answer(question: str, chunks: list[dict[str, Any]]) -> str:
    """
    Generates an answer based only on the provided context chunks.
    Includes inline citations back to the source documents.
    """
    if not chunks:
        return "I couldn't find relevant information in the uploaded document(s) to answer this."

    llm: BaseChatModel = get_llm(temperature=0.0)

    # Format the context
    context_blocks: list[str] = []
    for idx, c in enumerate(chunks):
        doc_name = c.get("doc_name", "Unknown")
        page_num = c.get("page_number", 1)
        text = c.get("chunk_text", "")
        context_blocks.append(
            f"Context [{idx + 1}]:\n"
            f"Source Document: {doc_name}\n"
            f"Page: {page_num}\n"
            f"Content: {text}\n"
        )
    context_str = "\n---\n".join(context_blocks)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful research assistant. Your task is to answer the user's question "
                    "based ONLY on the provided contexts. If the answer cannot be found or inferred from "
                    "the contexts, honestly state: 'I couldn't find relevant information in the uploaded document(s) to answer this.' "
                    "Do not make up facts or use external knowledge. "
                    "For statements you make that are based on a context block, include an inline citation "
                    "with its index in brackets, like 'according to the protocol [1]'. If multiple contexts "
                    "support a claim, include multiple citations like '[1][2]'. "
                    "Keep the response concise, clear, and strictly grounded in the context."
                ),
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_str, "question": question})

    return answer.strip()
