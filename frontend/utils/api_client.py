"""
API Client for backend communication.
Contains helper functions to interact with /health, /upload, and /ask FastAPI endpoints.
Includes retry logic, health checking, and structured error handling.
"""

import os
import time
import httpx

API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# ═══════════════════════════════════════════════════════════════════════════════
# Health Check
# ═══════════════════════════════════════════════════════════════════════════════

def check_backend_health(timeout: float = 5.0) -> bool:
    """
    Checks if the backend API is reachable and healthy.
    Returns True if the /health endpoint responds with 200 OK.
    """
    try:
        response = httpx.get(f"{API_URL}/health", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Document Upload
# ═══════════════════════════════════════════════════════════════════════════════

def upload_document_api(
    filename: str,
    content: bytes,
    session_id: str,
    max_retries: int = 2
) -> dict:
    """
    Uploads a document to the backend /upload endpoint with retry logic.

    Args:
        filename: Original filename of the document.
        content: Raw file bytes.
        session_id: Session identifier for associating the document.
        max_retries: Number of retry attempts on transient failures.

    Returns:
        dict with either success fields (doc_id, filename, chunks_created)
        or error fields (error, message).
    """
    url = f"{API_URL}/upload"
    files = {"file": (filename, content)}
    data = {"session_id": session_id}

    last_error = None
    for attempt in range(1, max_retries + 2):  # initial + retries
        try:
            response = httpx.post(
                url, files=files, data=data,
                timeout=120.0  # generous timeout for embedding large files
            )
            if response.status_code == 200:
                return response.json()
            else:
                # Server returned an error — don't retry client errors
                try:
                    err_data = response.json()
                    return {
                        "error": err_data.get("error", "upload_failed"),
                        "message": err_data.get("detail", err_data.get("message", "Upload failed"))
                    }
                except Exception:
                    return {
                        "error": "upload_failed",
                        "message": f"HTTP {response.status_code}: {response.text[:200]}"
                    }
        except httpx.TimeoutException:
            last_error = {
                "error": "timeout",
                "message": f"Upload timed out after 120s (attempt {attempt})"
            }
        except httpx.ConnectError:
            last_error = {
                "error": "connection_error",
                "message": f"Cannot reach backend at {url}. Is the server running?"
            }
            break  # no point retrying if we can't connect
        except Exception as e:
            last_error = {
                "error": "connection_error",
                "message": f"Connection failed: {str(e)}"
            }

        if attempt <= max_retries:
            time.sleep(1.5 * attempt)  # exponential-ish backoff

    return last_error or {"error": "unknown", "message": "Upload failed"}


# ═══════════════════════════════════════════════════════════════════════════════
# Question Answering
# ═══════════════════════════════════════════════════════════════════════════════

def ask_question_api(
    question: str,
    session_id: str,
    doc_ids: list[str] | None = None,
    max_retries: int = 1
) -> dict:
    """
    Sends a query to the backend /ask endpoint with retry logic.

    Args:
        question: The user's natural-language question.
        session_id: Session identifier for chat history context.
        doc_ids: Optional list of document IDs to scope the search.
        max_retries: Number of retry attempts on transient failures.

    Returns:
        dict with either success fields (answer, citations, agent_trace,
        hallucination_warning) or error fields (error, message).
    """
    url = f"{API_URL}/ask"
    payload = {
        "session_id": session_id,
        "question": question,
        "doc_ids": doc_ids
    }

    last_error = None
    for attempt in range(1, max_retries + 2):
        try:
            response = httpx.post(
                url, json=payload,
                timeout=90.0  # RAG pipeline can take time
            )
            if response.status_code == 200:
                return response.json()
            else:
                try:
                    err_data = response.json()
                    return {
                        "error": err_data.get("error", "ask_failed"),
                        "message": err_data.get("detail", err_data.get("message", "Request failed"))
                    }
                except Exception:
                    return {
                        "error": "ask_failed",
                        "message": f"HTTP {response.status_code}: {response.text[:200]}"
                    }
        except httpx.TimeoutException:
            last_error = {
                "error": "timeout",
                "message": "Request timed out. The RAG pipeline may be processing a large document."
            }
        except httpx.ConnectError:
            last_error = {
                "error": "connection_error",
                "message": f"Cannot reach backend at {url}. Is the server running?"
            }
            break
        except Exception as e:
            last_error = {
                "error": "connection_error",
                "message": f"Connection failed: {str(e)}"
            }

        if attempt <= max_retries:
            time.sleep(2.0 * attempt)

    return last_error or {"error": "unknown", "message": "Request failed"}
