"""
Integration test for LangGraph Agentic RAG pipeline.
Mocks LLM nodes if API keys are missing to verify execution flow and trace logs.
"""

import os
from pathlib import Path

import config

# Define mock functions first
def mock_rewrite_query(question, chat_history):
    print("[MOCK] Query Rewriter invoked")
    return f"standalone {question}"

def mock_grade_chunk_relevance(query, chunk_text):
    print(f"[MOCK] Relevance Grader evaluating: {chunk_text[:35]}...")
    return "ChromaDB" in chunk_text or "LangGraph" in chunk_text or "Query" in chunk_text

def mock_generate_answer(query, chunks):
    print("[MOCK] Answer Generator invoked")
    return "Mocked Agentic Answer: LangGraph and ChromaDB are used together, as cited in [1]."

def mock_check_hallucination(answer, chunks):
    print("[MOCK] Hallucination Checker invoked")
    return False

# Check if API keys are set. If not, mock the LLM agent functions.
has_key = False
if config.LLM_PROVIDER == "google" and config.GOOGLE_API_KEY and "your_google_api_key" not in config.GOOGLE_API_KEY:
    has_key = True
elif config.LLM_PROVIDER == "openai" and config.OPENAI_API_KEY and "your_openai_api_key" not in config.OPENAI_API_KEY:
    has_key = True

if not has_key:
    print("API key not set or placeholder. Mocking LLM agent modules for pipeline verification...")
    # Import and replace functions BEFORE importing the graph module
    import core.agents.query_rewriter
    core.agents.query_rewriter.rewrite_query = mock_rewrite_query
    
    import core.agents.relevance_grader
    core.agents.relevance_grader.grade_chunk_relevance = mock_grade_chunk_relevance
    
    import core.agents.answer_generator
    core.agents.answer_generator.generate_answer = mock_generate_answer
    
    import core.agents.hallucination_checker
    core.agents.hallucination_checker.check_hallucination = mock_check_hallucination
else:
    print(f"API key detected for provider: {config.LLM_PROVIDER}. Running actual agentic pipeline...")

# Now import graph and other modules
from core.agents.graph import run_agentic_rag
from core.ingestion import extract_text, chunk_document
from core.vector_store import VectorStoreManager

def setup_test_document():
    txt_path = Path("agentic_test.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(
            "LangGraph is a library for building stateful, multi-actor applications with LLMs.\n"
            "It extends LCEL with multi-agent coordination. "
            "ChromaDB is a vector store for semantic search.\n"
            "The Query Rewriter node reformulates raw inputs using prior history context.\n"
            "The Relevance Grader filters irrelevant retrieved chunks to prevent hallucination."
        )
    
    pages = extract_text(txt_path)
    chunks = chunk_document(pages, doc_id="doc_agentic_test", doc_name="agentic_test.txt")
    
    manager = VectorStoreManager()
    try:
        manager.delete_by_doc_id("doc_agentic_test")
    except Exception:
        pass
        
    manager.add_chunks(chunks)
    return txt_path

def main():
    txt_file = setup_test_document()
    
    try:
        chat_history = [
            {"role": "user", "content": "I want to ask about RAG framework libraries."},
            {"role": "assistant", "content": "Sure! I can help you with LangGraph and ChromaDB."}
        ]
        question = "What is LangGraph?"
        print(f"\nUser: {question}")
        
        result = run_agentic_rag(question, chat_history=chat_history, doc_ids=["doc_agentic_test"])
        
        print("\n--- Pipeline Results ---")
        print(f"Rewritten Query: {result.get('rewritten_query')}")
        print(f"Retrieved Chunks: {len(result.get('retrieved_chunks', []))}")
        print(f"Relevant Chunks (after Grader): {len(result.get('relevant_chunks', []))}")
        print(f"Final Answer: {result.get('answer')}")
        print(f"Agent Trace: {result.get('agent_trace')}")
        
        # Ensure trace contains all 5 nodes
        expected_trace = ["query_rewriter", "retriever", "relevance_grader", "answer_generator", "hallucination_checker"]
        assert result.get("agent_trace") == expected_trace, f"Expected trace {expected_trace}, got {result.get('agent_trace')}"
        print("\nPipeline trace matches specification perfectly!")
        
    finally:
        # Clean up
        if os.path.exists(txt_file):
            os.remove(txt_file)
            
        manager = VectorStoreManager()
        try:
            manager.delete_by_doc_id("doc_agentic_test")
        except Exception:
            pass

if __name__ == "__main__":
    main()
