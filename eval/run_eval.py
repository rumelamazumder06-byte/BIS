"""
BIS Standards RAG Engine - Evaluation Runner
Runs the RAG pipeline against the evaluation dataset and outputs metrics.

Usage:
    python eval/run_eval.py
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from src.ingestion.chunker import load_and_chunk_standards
from src.ingestion.embedder import BISVectorStore
from src.retrieval.hybrid_retriever import BM25Retriever, HybridRetriever
from src.retrieval.reranker import ReRanker, deduplicate_by_standard
from src.generation.llm_chain import RAGPipeline, GeminiLLM
from eval.eval_script import evaluate


def main():
    print("=" * 60)
    print("  BIS RAG Evaluation Runner")
    print("=" * 60)

    # 1. Initialize pipeline
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "bis_standards.json")
    chunks = load_and_chunk_standards(data_path)

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    vector_store = BISVectorStore(persist_dir=persist_dir, model_name=model_name)
    vector_store.add_chunks(chunks)

    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(vector_store, bm25)
    reranker = ReRanker()

    llm = None
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if api_key and api_key != "your_gemini_api_key_here":
        try:
            llm = GeminiLLM(api_key=api_key)
        except Exception:
            pass

    pipeline = RAGPipeline(hybrid, reranker, llm)

    # 2. Load eval dataset
    eval_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"\nRunning {len(eval_data)} evaluation queries...\n")

    # 3. Run queries
    results = []
    for entry in eval_data:
        query = entry["query"]
        relevant = entry["relevant_standards"]

        start = time.time()
        output = pipeline.recommend(product_description=query, top_k=5)
        latency = (time.time() - start) * 1000

        predictions = []
        for rec in output["recommendations"]:
            predictions.append({
                "standard_id": rec["standard_id"],
                "title": rec["title"],
            })

        results.append({
            "query_id": entry.get("query_id", 0),
            "query": query,
            "relevant_standards": relevant,
            "predictions": predictions,
            "latency_ms": round(latency, 2),
        })

        # Print progress
        pred_ids = [p["standard_id"] for p in predictions[:3]]
        hit = "HIT" if set(relevant).intersection(pred_ids) else "MISS"
        print(f"  {hit} Q{entry.get('query_id', '?')}: {query[:60]}...")
        print(f"     Expected: {relevant}")
        print(f"     Got top-3: {pred_ids}\n")

    # 4. Evaluate
    metrics = evaluate(results)

    print("\n" + "=" * 50)
    print("  FINAL EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Hit Rate @3:      {metrics['hit_rate_at_3']:.4f}  ({metrics['hit_rate_at_3']*100:.1f}%)")
    print(f"  MRR @5:           {metrics['mrr_at_5']:.4f}")
    print(f"  Avg Latency:      {metrics['avg_latency_ms']:.2f} ms")
    print(f"  Total Queries:    {metrics['total_queries']}")
    print("=" * 50)

    # 5. Save results
    output_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")

    metrics_path = os.path.join(os.path.dirname(__file__), "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
