"""
BIS Standards RAG Engine - Evaluation Script
Computes Hit Rate @3, MRR @5, and Latency from query-result pairs.

Usage:
    python eval_script.py --input eval_results.json
"""

import json
import argparse
import sys
from typing import List, Dict


def hit_rate_at_k(results: List[Dict], k: int = 3) -> float:
    """
    Hit Rate @K: Fraction of queries where at least one relevant standard
    appears in the top-K recommendations.
    """
    hits = 0
    for entry in results:
        relevant = set(entry["relevant_standards"])
        predicted = [r["standard_id"] for r in entry["predictions"][:k]]
        if relevant.intersection(predicted):
            hits += 1
    return hits / len(results) if results else 0.0


def mrr_at_k(results: List[Dict], k: int = 5) -> float:
    """
    Mean Reciprocal Rank @K: Average of 1/rank for the first relevant
    standard in the top-K predictions.
    """
    total_rr = 0.0
    for entry in results:
        relevant = set(entry["relevant_standards"])
        predicted = [r["standard_id"] for r in entry["predictions"][:k]]
        rr = 0.0
        for rank, pred_id in enumerate(predicted, 1):
            if pred_id in relevant:
                rr = 1.0 / rank
                break
        total_rr += rr
    return total_rr / len(results) if results else 0.0


def avg_latency(results: List[Dict]) -> float:
    """Average latency in milliseconds across all queries."""
    latencies = [entry.get("latency_ms", 0) for entry in results]
    return sum(latencies) / len(latencies) if latencies else 0.0


def evaluate(results: List[Dict]) -> Dict:
    """Run all evaluation metrics."""
    metrics = {
        "hit_rate_at_3": round(hit_rate_at_k(results, k=3), 4),
        "mrr_at_5": round(mrr_at_k(results, k=5), 4),
        "avg_latency_ms": round(avg_latency(results), 2),
        "total_queries": len(results),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="BIS RAG Evaluation Script")
    parser.add_argument("--input", type=str, required=True, help="Path to JSON file with query-result pairs")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    metrics = evaluate(results)

    print("\n" + "=" * 50)
    print("  BIS RAG Evaluation Results")
    print("=" * 50)
    print(f"  Total Queries:    {metrics['total_queries']}")
    print(f"  Hit Rate @3:      {metrics['hit_rate_at_3']:.4f}")
    print(f"  MRR @5:           {metrics['mrr_at_5']:.4f}")
    print(f"  Avg Latency (ms): {metrics['avg_latency_ms']:.2f}")
    print("=" * 50)

    # Save results
    output_path = args.input.replace(".json", "_metrics.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {output_path}")

    return metrics


if __name__ == "__main__":
    main()
