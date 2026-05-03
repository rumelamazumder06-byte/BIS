"""
BIS Standards RAG Engine - Re-Ranker Module
Uses a cross-encoder model to re-rank retrieved candidates for precision.
"""

from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder


class ReRanker:
    """Cross-encoder based re-ranker for improving retrieval precision."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading re-ranker model: {model_name}...")
        self.model = CrossEncoder(model_name, max_length=512)
        print("Re-ranker ready.")

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Re-rank documents using cross-encoder scoring.
        
        The cross-encoder jointly encodes (query, document) pairs for
        more accurate relevance scoring than bi-encoder similarity.
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [(query, doc["text"]) for doc in documents]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Attach scores and sort
        scored_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored_docs[:top_k]


def deduplicate_by_standard(documents: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Deduplicate results to show at most one chunk per BIS standard.
    Keeps the highest-scoring chunk for each standard_id.
    """
    seen_standards = set()
    unique_docs = []

    for doc in documents:
        std_id = doc["metadata"]["standard_id"]
        if std_id not in seen_standards:
            seen_standards.add(std_id)
            unique_docs.append(doc)
            if len(unique_docs) >= top_k:
                break

    return unique_docs
