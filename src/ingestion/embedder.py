"""
BIS Standards RAG Engine - Embedding & Vector Store Module
Embeds document chunks and stores them in ChromaDB for dense retrieval.
"""

import os
import json
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class BISVectorStore:
    """Manages ChromaDB vector store for BIS standard chunks."""

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "bis_standards",
    ):
        self.model_name = model_name
        self.collection_name = collection_name
        self.persist_dir = persist_dir

        # Load sentence-transformer model
        print(f"Loading embedding model: {model_name}...")
        self.embed_model = SentenceTransformer(model_name)

        # Initialize ChromaDB client with persistence
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"ChromaDB collection '{collection_name}' ready. Documents: {self.collection.count()}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.embed_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        return embeddings.tolist()

    def add_chunks(self, chunks: List[Dict], batch_size: int = 50):
        """Add document chunks to ChromaDB. Skips if already populated."""
        if self.collection.count() > 0:
            print(f"Collection already has {self.collection.count()} documents. Skipping ingestion.")
            return

        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [
            {
                "standard_id": c["standard_id"],
                "title": c["title"],
                "category": c["category"],
                "level": c["level"],
                "keywords": ", ".join(c.get("keywords", [])),
            }
            for c in chunks
        ]

        # Batch embed and add
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]
            batch_embeddings = self.embed_texts(batch_texts)

            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metas,
            )
            print(f"  Added batch {i // batch_size + 1} ({len(batch_texts)} chunks)")

        print(f"Total documents in store: {self.collection.count()}")

    def search(self, query: str, top_k: int = 10, category_filter: Optional[str] = None) -> List[Dict]:
        """Dense similarity search over the vector store."""
        query_embedding = self.embed_texts([query])[0]

        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # cosine distance → similarity
            })
        return docs

    def reset(self):
        """Delete and recreate the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print("Collection reset.")


if __name__ == "__main__":
    from src.ingestion.chunker import load_and_chunk_standards

    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "bis_standards.json")
    chunks = load_and_chunk_standards(data_path)

    store = BISVectorStore(persist_dir="./data/chroma_db")
    store.add_chunks(chunks)

    # Test search
    results = store.search("high strength cement for bridge construction", top_k=5)
    print("\n=== Search Results ===")
    for r in results:
        print(f"  [{r['score']:.3f}] {r['metadata']['standard_id']} — {r['metadata']['title']}")
