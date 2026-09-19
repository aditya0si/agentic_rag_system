"""
Integration tests for FastAPI endpoints (/upload, /ask) using TestClient.
Mocks LLM nodes to test pipeline execution traces and response structures.
"""


# Define mocks first
def mock_rewrite_query(question, chat_history):
    return f"standalone {question}"


def mock_grade_chunk_relevance(query, chunk_text):
    return True


def mock_generate_answer(query, chunks):
    return "Mocked Answer: LangGraph coordinates multiple chains, as cited in [1]."


# Mock functions BEFORE importing FastAPI application so graph binds to mocked references
import core.agents.query_rewriter

core.agents.query_rewriter.rewrite_query = mock_rewrite_query

import core.agents.relevance_grader

core.agents.relevance_grader.grade_chunk_relevance = mock_grade_chunk_relevance

import core.agents.answer_generator

core.agents.answer_generator.generate_answer = mock_generate_answer


def mock_check_hallucination(answer, chunks):
    return False


import core.agents.hallucination_checker

core.agents.hallucination_checker.check_hallucination = mock_check_hallucination

from fastapi.testclient import TestClient
from main import app


def test_endpoints():
    client = TestClient(app)
    session_id = "test_session_123"

    print("\n--- 1. Testing Upload of Unsupported File Type ---")
    files = {"file": ("test.py", b"print('hello')", "text/plain")}
    response = client.post("/upload", files=files, data={"session_id": session_id})
    print("Status code:", response.status_code)
    print("Response:", response.json())
    assert response.status_code == 422
    assert response.json()["error"] == "unsupported_file_type"

    print("\n--- 2. Testing Ask Question before uploading ---")
    ask_payload = {"session_id": "empty_session", "question": "What is LangGraph?"}
    response = client.post("/ask", json=ask_payload)
    print("Status code:", response.status_code)
    print("Response:", response.json())
    assert response.status_code == 200
    assert "upload at least one document" in response.json()["answer"]

    print("\n--- 3. Testing Document Upload (TXT) ---")
    txt_content = (
        "LangGraph is a library for building stateful, multi-actor applications with LLMs.\n"
        "It extends LCEL with multi-agent coordination capability.\n"
    )
    files = {"file": ("test_doc.txt", txt_content.encode("utf-8"), "text/plain")}
    response = client.post("/upload", files=files, data={"session_id": session_id})
    print("Status code:", response.status_code)
    print("Response:", response.json())
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "indexed"
    doc_id = res_data["doc_id"]

    print("\n--- 4. Testing Ask Question (Agentic Graph) ---")
    ask_payload = {"session_id": session_id, "question": "What is LangGraph?"}
    response = client.post("/ask", json=ask_payload)
    print("Status code:", response.status_code)
    print("Response:", response.json())
    assert response.status_code == 200
    res_data = response.json()
    assert "LangGraph" in res_data["answer"]
    assert len(res_data["citations"]) > 0
    assert res_data["citations"][0]["doc_id"] == doc_id
    assert res_data["citations"][0]["doc_name"] == "test_doc.txt"
    # Verify the agent trace contains all 4 nodes
    expected_trace = [
        "query_rewriter",
        "retriever",
        "relevance_grader",
        "answer_generator",
        "hallucination_checker",
    ]
    assert res_data["agent_trace"] == expected_trace, (
        f"Expected {expected_trace}, got {res_data['agent_trace']}"
    )
    print("Agent trace verified:", res_data["agent_trace"])

    # Clean up ChromaDB collection
    from core.vector_store import VectorStoreManager

    manager = VectorStoreManager()
    manager.delete_by_doc_id(doc_id)
    print("\nCleanup completed. All tests passed successfully!")


if __name__ == "__main__":
    test_endpoints()
