"""
Session and memory store — maintains in-memory chat history and uploaded document IDs per session.
"""

from typing import Dict, List, Any, TypedDict


class SessionData(TypedDict):
    """Session data structure."""
    doc_ids: List[str]
    chat_history: List[Dict[str, str]]


# Simple in-memory storage:
# {
#     session_id: {
#         "doc_ids": [doc_id1, doc_id2, ...],
#         "chat_history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
#     }
# }
_sessions_store: Dict[str, SessionData] = {}


def get_or_create_session(session_id: str) -> SessionData:
    """
    Retrieves a session or initializes it if it does not exist.
    """
    if session_id not in _sessions_store:
        _sessions_store[session_id] = {
            "doc_ids": [],
            "chat_history": []
        }
    return _sessions_store[session_id]


def add_document_to_session(session_id: str, doc_id: str) -> None:
    """
    Associates a document ID with a session.
    """
    session = get_or_create_session(session_id)
    if doc_id not in session["doc_ids"]:
        session["doc_ids"].append(doc_id)


def get_session_documents(session_id: str) -> List[str]:
    """
    Returns all document IDs uploaded in the given session.
    """
    session = get_or_create_session(session_id)
    return session["doc_ids"]


def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    """
    Returns the chat history for a session.
    """
    session = get_or_create_session(session_id)
    return session["chat_history"]


def add_message_to_history(session_id: str, role: str, content: str) -> None:
    """
    Appends a message (user or assistant) to the session's chat history.
    """
    session = get_or_create_session(session_id)
    session["chat_history"].append({"role": role, "content": content})


def clear_session(session_id: str) -> None:
    """
    Clears all chat history and document links for a session.
    """
    if session_id in _sessions_store:
        del _sessions_store[session_id]
