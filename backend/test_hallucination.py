"""
Integration test for the LangGraph Agentic RAG pipeline's Hallucination Checker.
Verify the hallucination node triggers and correctly sets warning messages.
"""

import os
from pathlib import Path

import config

# Define mock functions for Query Rewriter, Relevance Grader, and Answer Generator
def mock_rewrite_query(question, chat_history):
    return f"standalone {question}"

def mock_grade_chunk_relevance(query, chunk_text):
    return True

def mock_generate_answer(query, chunks):
    return "Mocked Answer about LangGraph"

# For testing hallucination, we want to mock check_hallucination to return True (hallucination detected)
def mock_check_hallucination_yes(answer, chunks):
    print("[MOCK] Hallucination Checker invoked: returning YES (hallucination detected)")
    return True

# For testing no hallucination, we want check_hallucination to return False (grounded)
def mock_check_hallucination_no(answer, chunks):
    print("[MOCK] Hallucination Checker invoked: returning NO (grounded)")
    return False

# Setup mocking before importing graph
has_key = False
if config.LLM_PROVIDER == "google" and config.GOOGLE_API_KEY and "your_google_api_key" not in config.GOOGLE_API_KEY:
    has_key = True
elif config.LLM_PROVIDER == "openai" and config.OPENAI_API_KEY and "your_openai_api_key" not in config.OPENAI_API_KEY:
    has_key = True

if not has_key:
    print("API key not set. Mocking all LLM agent modules...")
    import core.agents.query_rewriter
    core.agents.query_rewriter.rewrite_query = mock_rewrite_query
    
    import core.agents.relevance_grader
    core.agents.relevance_grader.grade_chunk_relevance = mock_grade_chunk_relevance
    
    import core.agents.answer_generator
    core.agents.answer_generator.generate_answer = mock_generate_answer
    
    # We will dynamically override check_hallucination in our test cases
    import core.agents.hallucination_checker
else:
    print(f"API key detected. Running real agentic pipeline with LLM...")

# Import graph
from core.agents.graph import run_agentic_rag
from core.ingestion import extract_text, chunk_document
from core.vector_store import VectorStoreManager

def setup_test_document():
    txt_path = Path("hallucination_test.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("LangGraph is a library for building stateful, multi-actor applications with LLMs.")
    
    pages = extract_text(txt_path)
    chunks = chunk_document(pages, doc_id="doc_hal_test", doc_name="hallucination_test.txt")
    
    manager = VectorStoreManager()
    try:
        manager.delete_by_doc_id("doc_hal_test")
    except Exception:
        pass
    manager.add_chunks(chunks)
    return txt_path

def test_hallucination_flow():
    txt_file = setup_test_document()
    manager = VectorStoreManager()
    
    try:
        # Override the check_hallucination function in the graph module to simulate hallucination
        import core.agents.graph
        original_check = core.agents.graph.check_hallucination
        
        # Test Case 1: Hallucination is detected
        print("\n--- Test Case 1: Hallucination Detected ---")
        core.agents.graph.check_hallucination = mock_check_hallucination_yes
        
        result1 = run_agentic_rag("What is LangGraph?", chat_history=[], doc_ids=["doc_hal_test"])
        print(f"Agent Trace: {result1.get('agent_trace')}")
        print(f"Warning: {result1.get('hallucination_warning')}")
        
        assert "hallucination_checker" in result1.get("agent_trace")
        assert result1.get("hallucination_warning") is not None
        assert "Some claims may not be fully supported" in result1.get("hallucination_warning")
        print("Test Case 1 Passed!")
        
        # Test Case 2: Answer is fully grounded (No hallucination)
        print("\n--- Test Case 2: Answer Grounded (No Hallucination) ---")
        core.agents.graph.check_hallucination = mock_check_hallucination_no
        
        result2 = run_agentic_rag("What is LangGraph?", chat_history=[], doc_ids=["doc_hal_test"])
        print(f"Agent Trace: {result2.get('agent_trace')}")
        print(f"Warning: {result2.get('hallucination_warning')}")
        
        assert "hallucination_checker" in result2.get("agent_trace")
        assert result2.get("hallucination_warning") is None
        print("Test Case 2 Passed!")
        
        # Restore original
        core.agents.graph.check_hallucination = original_check
        
    finally:
        # Cleanup
        if os.path.exists(txt_file):
            os.remove(txt_file)
        try:
            manager.delete_by_doc_id("doc_hal_test")
        except Exception:
            pass

if __name__ == "__main__":
    test_hallucination_flow()
