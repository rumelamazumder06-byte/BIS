"""
BIS Standards RAG Engine - Hybrid Retriever
Combines dense (ChromaDB) + sparse (BM25) retrieval with Reciprocal Rank Fusion.
"""

import re
from typing import List, Dict, Optional, Tuple
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Sparse retriever using BM25 algorithm for keyword-based matching."""

    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        # Tokenize documents for BM25
        self.tokenized_docs = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"BM25 index built with {len(chunks)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        return tokens

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search using BM25 scoring."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in scored_indices:
            if score > 0:
                results.append({
                    "chunk_id": self.chunks[idx]["chunk_id"],
                    "text": self.chunks[idx]["text"],
                    "metadata": {
                        "standard_id": self.chunks[idx]["standard_id"],
                        "title": self.chunks[idx]["title"],
                        "category": self.chunks[idx]["category"],
                        "level": self.chunks[idx]["level"],
                        "keywords": ", ".join(self.chunks[idx].get("keywords", [])),
                    },
                    "score": float(score),
                })
        return results


def reciprocal_rank_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict]:
    """
    Merge dense and sparse retrieval results using Reciprocal Rank Fusion (RRF).
    
    RRF score = sum( weight / (k + rank) ) for each result list.
    This is a proven fusion method that handles different score scales.
    """
    fused_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    # Score dense results
    for rank, doc in enumerate(dense_results):
        doc_id = doc["chunk_id"]
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + dense_weight / (k + rank + 1)
        doc_map[doc_id] = doc

    # Score sparse results
    for rank, doc in enumerate(sparse_results):
        doc_id = doc["chunk_id"]
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + sparse_weight / (k + rank + 1)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    # Sort by fused score
    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    fused_results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].copy()
        doc["fused_score"] = fused_scores[doc_id]
        fused_results.append(doc)

    return fused_results


class HybridRetriever:
    """
    Hybrid retriever combining dense vector search (ChromaDB) with
    sparse BM25 search, merged via Reciprocal Rank Fusion.
    """

    def __init__(self, vector_store, bm25_retriever: BM25Retriever, dense_weight=0.6, sparse_weight=0.4):
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve(self, query: str, top_k: int = 10, category_filter: Optional[str] = None) -> List[Dict]:
        """
        Retrieve documents using hybrid dense + sparse search with RRF.
        """
        # Dense retrieval from ChromaDB
        dense_results = self.vector_store.search(query, top_k=top_k * 2, category_filter=category_filter)

        # Sparse retrieval from BM25
        sparse_results = self.bm25.search(query, top_k=top_k * 2)

        # Fuse results
        fused = reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
        )

        return fused[:top_k]
