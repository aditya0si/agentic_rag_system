"""
Integration test for Basic RAG Chain.
Sets up a test document, runs query, and outputs answer + citation structure.
Mocks LLM call if API keys are not configured.
"""

import os
from pathlib import Path

import config
from core.basic_rag import run_basic_rag
from core.ingestion import chunk_document, extract_text
from core.vector_store import VectorStoreManager


def setup_test_document():
    txt_path = Path("rag_test.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(
            "LangGraph is a library for building stateful, multi-actor applications with LLMs.\n"
            "It extends the LangChain Expression Language (LCEL) with the ability to coordinate multiple chains.\n"
            "ChromaDB is a vector database used to store and query document embeddings.\n"
            "A standard RAG pipeline retrieves relevant chunks and feeds them to an LLM."
        )

    pages = extract_text(txt_path)
    chunks = chunk_document(pages, doc_id="doc_rag_test", doc_name="rag_test.txt")

    manager = VectorStoreManager()
    try:
        manager.delete_by_doc_id("doc_rag_test")
    except Exception:
        pass

    manager.add_chunks(chunks)
    return txt_path


def main():
    txt_file = setup_test_document()

    print("\n--- Testing Basic RAG Chain ---")
    question = "What is LangGraph?"

    # Check if API keys are set. If not, mock the LLM call to verify the chain execution.
    has_key = False
    if (
        config.LLM_PROVIDER == "google"
        and config.GOOGLE_API_KEY
        and "your_google_api_key" not in config.GOOGLE_API_KEY
    ):
        has_key = True
    elif (
        config.LLM_PROVIDER == "openai"
        and config.OPENAI_API_KEY
        and "your_openai_api_key" not in config.OPENAI_API_KEY
    ):
        has_key = True

    if not has_key:
        print("API key not set or placeholder. Mocking LLM response to test pipeline routing...")
        # Mock LLM invocation in basic_rag directly since it already imported it
        import core.basic_rag

        def mock_generate_answer(q, chunks):
            return "Mocked Answer: LangGraph is a library for building stateful, multi-actor applications with LLMs, as cited in [1]."

        core.basic_rag.generate_answer = mock_generate_answer
    else:
        print(f"API key detected for provider: {config.LLM_PROVIDER}. Running actual LLM query...")

    try:
        response = run_basic_rag(question, doc_ids=["doc_rag_test"])
        print("\nPipeline Response:")
        print(f"Answer: {response.answer}")
        print("Citations:")
        for citation in response.citations:
            print(
                f"  - Doc: {citation.doc_name}, Page: {citation.page}, Text snippet: {citation.chunk_text[:100]}..."
            )
        print(f"Agent Trace: {response.agent_trace}")

    finally:
        # Clean up
        if os.path.exists(txt_file):
            os.remove(txt_file)

        manager = VectorStoreManager()
        try:
            manager.delete_by_doc_id("doc_rag_test")
        except Exception:
            pass


if __name__ == "__main__":
    main()
