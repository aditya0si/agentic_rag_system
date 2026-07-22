"""
RAGAS Evaluation Script - Compare Naive RAG vs Agentic RAG.

Runs both pipelines against a test set and computes RAGAS metrics:
- Faithfulness: How factually consistent the answer is with the context
- Answer Relevancy: How relevant the answer is to the question
- Context Precision: How relevant the retrieved context is to the question
- Context Recall: How much of the relevant context was retrieved
"""

import json
import os
import sys
from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.basic_rag import run_basic_rag
from core.agents.graph import run_agentic_rag
from core.ingestion import extract_text, chunk_document
from core.vector_store import VectorStoreManager


@dataclass
class EvaluationResult:
    """Results for a single evaluation."""
    question: str
    ground_truth: str
    naive_answer: str
    agentic_answer: str
    naive_contexts: list[str]
    agentic_contexts: list[str]
    metrics: dict[str, float]


def setup_test_documents() -> list[str]:
    """Create test documents and return doc_ids."""
    test_docs = [
        {
            "doc_id": "eval_doc_1",
            "doc_name": "langgraph_overview.txt",
            "content": (
                "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
                "It extends the LangChain Expression Language (LCEL) with the ability to coordinate multiple chains. "
                "LangGraph provides a graph-based approach to orchestrating LLM workflows with nodes and edges. "
                "It supports cycles, branching, and parallel execution."
            )
        },
        {
            "doc_id": "eval_doc_2",
            "doc_name": "rag_pipeline.txt",
            "content": (
                "The agentic RAG pipeline consists of 5 agents: Query Rewriter, Retriever, "
                "Relevance Grader, Answer Generator, and Hallucination Checker. "
                "The Query Rewriter reformulates questions using chat history. "
                "The Retriever fetches top-k chunks from ChromaDB. "
                "The Relevance Grader filters irrelevant chunks. "
                "The Answer Generator produces grounded responses with citations. "
                "The Hallucination Checker validates answers against sources."
            )
        },
        {
            "doc_id": "eval_doc_3",
            "doc_name": "technical_details.txt",
            "content": (
                "ChromaDB is a vector database for semantic search that stores document embeddings. "
                "The system uses all-MiniLM-L6-v2 for free local embeddings via HuggingFace. "
                "Documents are chunked with RecursiveCharacterTextSplitter at 500 chars with 50 overlap. "
                "Supported file types: PDF, DOCX, TXT, MD with 20MB size limit. "
                "Google Gemini (gemini-2.0-flash) is the default LLM provider."
            )
        }
    ]
    
    manager = VectorStoreManager()
    doc_ids = []
    
    for doc in test_docs:
        # Clean up any existing
        try:
            manager.delete_by_doc_id(doc["doc_id"])
        except Exception:
            pass
        
        # Create temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(doc["content"])
            temp_path = f.name
        
        # Extract and chunk
        pages = extract_text(temp_path)
        chunks = chunk_document(pages, doc_id=doc["doc_id"], doc_name=doc["doc_name"])
        
        # Add to vector store
        manager.add_chunks(chunks)
        doc_ids.append(doc["doc_id"])
        
        # Clean up temp file
        os.unlink(temp_path)
    
    return doc_ids


def cleanup_test_documents(doc_ids: list[str]) -> None:
    """Remove test documents from vector store."""
    manager = VectorStoreManager()
    for doc_id in doc_ids:
        try:
            manager.delete_by_doc_id(doc_id)
        except Exception:
            pass


def evaluate_naive_rag(questions: list[str], doc_ids: list[str]) -> list[dict[str, Any]]:
    """Run naive RAG on all questions."""
    results = []
    for q in questions:
        response = run_basic_rag(q, doc_ids=doc_ids)
        # Get contexts used
        from core.agents.retriever import retrieve_chunks
        chunks = retrieve_chunks(q, doc_ids=doc_ids)
        contexts = [c["chunk_text"] for c in chunks]
        
        results.append({
            "question": q,
            "answer": response.answer,
            "contexts": contexts,
            "citations": [
                {
                    "doc_id": c.doc_id,
                    "doc_name": c.doc_name,
                    "page": c.page,
                    "text": c.chunk_text
                }
                for c in response.citations
            ]
        })
    return results


def evaluate_agentic_rag(questions: list[str], doc_ids: list[str]) -> list[dict[str, Any]]:
    """Run agentic RAG on all questions."""
    results = []
    for q in questions:
        result = run_agentic_rag(q, chat_history=[], doc_ids=doc_ids)
        contexts = [c["chunk_text"] for c in result.get("relevant_chunks", [])]
        
        results.append({
            "question": q,
            "answer": result.get("answer", ""),
            "contexts": contexts,
            "agent_trace": result.get("agent_trace", []),
            "hallucination_warning": result.get("hallucination_warning")
        })
    return results


def compute_ragas_metrics(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str]
) -> dict[str, float]:
    """
    Compute RAGAS metrics using the RAGAS library if available,
    otherwise compute simplified versions.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
        
        # Prepare dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data)
        
        # Run evaluation
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
        
        return {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "context_precision": float(result["context_precision"]),
            "context_recall": float(result["context_recall"]),
        }
    except ImportError:
        # Fallback: simplified metrics
        print("RAGAS not installed, computing simplified metrics...")
        return compute_simplified_metrics(questions, answers, contexts, ground_truths)


def compute_simplified_metrics(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str]
) -> dict[str, float]:
    """Compute simplified metrics when RAGAS is not available."""
    from difflib import SequenceMatcher
    
    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    recall_scores = []
    
    for q, ans, ctxs, gt in zip(questions, answers, contexts, ground_truths):
        # Faithfulness: how much answer is supported by contexts
        ctx_text = " ".join(ctxs)
        faithfulness_scores.append(similarity(ans, ctx_text) if ctx_text else 0.0)
        
        # Answer Relevancy: how relevant answer is to question
        relevancy_scores.append(similarity(ans, q))
        
        # Context Precision: how relevant contexts are to question
        if ctxs:
            ctx_relevance = [similarity(ctx, q) for ctx in ctxs]
            precision_scores.append(sum(ctx_relevance) / len(ctx_relevance))
        else:
            precision_scores.append(0.0)
        
        # Context Recall: how much ground truth is covered by contexts
        if ctxs:
            gt_coverage = [similarity(ctx, gt) for ctx in ctxs]
            recall_scores.append(max(gt_coverage) if gt_coverage else 0.0)
        else:
            recall_scores.append(0.0)
    
    return {
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
        "answer_relevancy": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0,
        "context_precision": sum(precision_scores) / len(precision_scores) if precision_scores else 0.0,
        "context_recall": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
    }


def generate_comparison_table(
    naive_metrics: dict[str, float],
    agentic_metrics: dict[str, float]
) -> str:
    """Generate a markdown comparison table."""
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    
    table = "| Metric | Naive RAG | Agentic RAG | Improvement |\n"
    table += "|--------|-----------|-------------|-------------|\n"
    
    for metric in metrics:
        naive = naive_metrics.get(metric, 0)
        agentic = agentic_metrics.get(metric, 0)
        improvement = ((agentic - naive) / naive * 100) if naive > 0 else 0
        table += f"| {metric.replace('_', ' ').title()} | {naive:.3f} | {agentic:.3f} | {improvement:+.1f}% |\n"
    
    return table


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("RAGAS Evaluation: Naive RAG vs Agentic RAG")
    print("=" * 60)
    
    # Load test set
    test_set_path = Path(__file__).parent / "test_set.json"
    with open(test_set_path, "r") as f:
        test_set = json.load(f)
    
    questions = [item["question"] for item in test_set]
    ground_truths = [item["ground_truth"] for item in test_set]
    
    # Setup test documents
    print("\nSetting up test documents...")
    doc_ids = setup_test_documents()
    print(f"Created {len(doc_ids)} test documents")
    
    try:
        # Evaluate Naive RAG
        print("\n--- Evaluating Naive RAG ---")
        naive_results = evaluate_naive_rag(questions, doc_ids)
        naive_answers = [r["answer"] for r in naive_results]
        naive_contexts = [r["contexts"] for r in naive_results]
        
        print(f"Evaluated {len(naive_results)} questions")
        
        # Evaluate Agentic RAG
        print("\n--- Evaluating Agentic RAG ---")
        agentic_results = evaluate_agentic_rag(questions, doc_ids)
        agentic_answers = [r["answer"] for r in agentic_results]
        agentic_contexts = [r["contexts"] for r in agentic_results]
        
        print(f"Evaluated {len(agentic_results)} questions")
        
        # Compute metrics
        print("\n--- Computing Metrics ---")
        print("\nNaive RAG Metrics:")
        naive_metrics = compute_ragas_metrics(
            questions, naive_answers, naive_contexts, ground_truths
        )
        for k, v in naive_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        print("\nAgentic RAG Metrics:")
        agentic_metrics = compute_ragas_metrics(
            questions, agentic_answers, agentic_contexts, ground_truths
        )
        for k, v in agentic_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Generate comparison
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(generate_comparison_table(naive_metrics, agentic_metrics))
        
        # Save detailed results
        output = {
            "naive_rag": {
                "metrics": naive_metrics,
                "results": naive_results
            },
            "agentic_rag": {
                "metrics": agentic_metrics,
                "results": agentic_results
            },
            "test_set": test_set
        }
        
        output_path = Path(__file__).parent / "evaluation_results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {output_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            naive = naive_metrics.get(metric, 0)
            agentic = agentic_metrics.get(metric, 0)
            winner = "Agentic" if agentic > naive else "Naive" if naive > agentic else "Tie"
            print(f"{metric}: {winner} wins (Naive: {naive:.3f}, Agentic: {agentic:.3f})")
        
    finally:
        # Cleanup
        print("\nCleaning up test documents...")
        cleanup_test_documents(doc_ids)
        print("Done!")


if __name__ == "__main__":
    main()