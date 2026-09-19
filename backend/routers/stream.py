"""
Streaming question answering endpoint — Server-Sent Events (SSE).

Streams agent pipeline progress events to the client in real time,
so the UI can show live status updates per agent node.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..core.agents.graph import AgentState, agentic_pipeline
from ..core.memory import add_message_to_history, get_chat_history, get_session_documents
from ..models.schemas import AskRequest

router = APIRouter()


@router.post("/ask/stream")
async def ask_question_stream(request: AskRequest) -> StreamingResponse:
    """
    Streams agent pipeline progress via Server-Sent Events.

    Emits JSON events:
      - {"type": "node_start", "node": "..."}
      - {"type": "node_end", "node": "...", "latency_s": ...}
      - {"type": "answer", "content": "...", "citations": [...], "hallucination_warning": ...}
      - {"type": "error", "message": "..."}
    """
    doc_ids = request.doc_ids or get_session_documents(request.session_id)

    if not doc_ids:

        async def no_docs() -> AsyncIterator[str]:
            yield f"data: {json.dumps({'type': 'answer', 'content': 'Please upload at least one document before asking questions.', 'citations': [], 'hallucination_warning': None})}\n\n"

        return StreamingResponse(no_docs(), media_type="text/event-stream")

    chat_history = get_chat_history(request.session_id)
    add_message_to_history(request.session_id, "user", request.question)

    initial_state: AgentState = {
        "question": request.question,
        "doc_ids": doc_ids,
        "chat_history": chat_history,
        "rewritten_query": "",
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "answer": "",
        "agent_trace": [],
        "hallucination_warning": None,
        "node_latencies": {},
    }

    async def event_generator() -> AsyncIterator[str]:
        try:
            # `astream` executes the graph already. Build the final state from
            # its node updates instead of invoking the graph a second time.
            # The old implementation doubled LLM calls, cost, and latency.
            final_state: dict[str, Any] = dict(initial_state)
            # Use LangGraph's streaming API to get per-node updates
            async for event in agentic_pipeline.astream(initial_state, stream_mode="updates"):
                for node_name, node_output in event.items():
                    final_state.update(node_output)
                    yield f"data: {json.dumps({'type': 'node_end', 'node': node_name, 'keys': list(node_output.keys())})}\n\n"
                    await asyncio.sleep(0)  # yield control

            answer = final_state.get("answer", "")
            relevant_chunks = final_state.get("relevant_chunks", [])
            citations: list[dict[str, Any]] = []
            if relevant_chunks and "I couldn't find relevant information" not in answer:
                for c in relevant_chunks:
                    citations.append(
                        {
                            "doc_id": c["doc_id"],
                            "doc_name": c.get("doc_name", ""),
                            "page": c.get("page_number", 1),
                            "chunk_text": c["chunk_text"],
                        }
                    )

            add_message_to_history(request.session_id, "assistant", answer)

            yield f"data: {json.dumps({'type': 'answer', 'content': answer, 'citations': citations, 'hallucination_warning': final_state.get('hallucination_warning'), 'agent_trace': final_state.get('agent_trace', []), 'node_latencies': final_state.get('node_latencies', {})})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
