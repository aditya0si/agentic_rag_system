"""
Question answering endpoint — accepts a question, runs RAG, and manages history.
"""

from fastapi import APIRouter, HTTPException, status

from ..core.agents.graph import run_agentic_rag
from ..core.memory import add_message_to_history, get_chat_history, get_session_documents
from ..core.security import detect_prompt_injection, sanitize_chat_history, validate_question
from ..models.schemas import AskRequest, AskResponse, Citation

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Submits a question. Retrieves relevant documents, runs RAG,
    logs the QA turn to session memory, and returns the answer with citations.
    """
    # 0. Validate and sanitize user input
    try:
        question = validate_question(request.question)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e

    # 0a. Heuristic prompt-injection guard
    is_suspicious, _ = detect_prompt_injection(question)
    # We log but do not block — the grounded generation prompt is robust enough
    # to ignore injection attempts. Blocking would hurt UX for edge-case phrasing.

    # 1. Determine which document IDs to query
    doc_ids = request.doc_ids
    if not doc_ids:
        doc_ids = get_session_documents(request.session_id)

    # If no documents are associated with the session, ask the user to upload one first.
    if not doc_ids:
        return AskResponse(
            answer="Please upload at least one document before asking questions.",
            citations=[],
            agent_trace=[],
        )

    # 2. Get chat history before adding the new message (sanitized)
    chat_history = sanitize_chat_history(get_chat_history(request.session_id))

    # Add user question to session history
    add_message_to_history(request.session_id, "user", question)

    try:
        # 3. Execute Agentic RAG pipeline
        result = run_agentic_rag(question, chat_history=chat_history, doc_ids=doc_ids)

        # Convert relevant chunks to citations
        citations: list[Citation] = []
        answer = result.get("answer", "")
        relevant_chunks = result.get("relevant_chunks", [])

        if relevant_chunks and "I couldn't find relevant information" not in answer:
            for c in relevant_chunks:
                citations.append(
                    Citation(
                        doc_id=c["doc_id"],
                        doc_name=c.get("doc_name", ""),
                        page=c.get("page_number", 1),
                        chunk_text=c["chunk_text"],
                    )
                )

        # 4. Add generated answer to session history
        add_message_to_history(request.session_id, "assistant", answer)

        return AskResponse(
            answer=answer,
            citations=citations,
            agent_trace=result.get("agent_trace", []),
            hallucination_warning=result.get("hallucination_warning"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your question: {str(e)}",
        ) from e
