"""
BIS Standards RAG Engine - FastAPI Application
Production-grade REST API for BIS standard recommendations.
"""

import os
import sys
import json
import time
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bis-rag")

from src.ingestion.chunker import load_and_chunk_standards
from src.ingestion.embedder import BISVectorStore
from src.retrieval.hybrid_retriever import BM25Retriever, HybridRetriever
from src.retrieval.reranker import ReRanker
from src.generation.llm_chain import RAGPipeline, _get_llm


# --- Pydantic Models ---

class ProductRequest(BaseModel):
    product_description: str = Field(..., min_length=3, description="Description of the product")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of standards to recommend")
    category_filter: Optional[str] = Field(default=None, description="Filter: Cement, Steel, Concrete, Aggregates")


class RecommendationItem(BaseModel):
    rank: int
    standard_id: str
    title: str
    rationale: str
    confidence: str
    key_clauses: List[str] = []


class SourceDocument(BaseModel):
    chunk_id: str = ""
    standard_id: str
    title: str
    category: str
    score: float = 0.0
    text_preview: str = ""


class RecommendationResponse(BaseModel):
    product_description: str
    recommendations: List[RecommendationItem]
    latency_ms: float
    source_chunks_count: int
    method: str
    source_documents: List[SourceDocument] = []


class HealthResponse(BaseModel):
    status: str
    engine: str
    standards_count: int
    chunks_count: int
    llm_mode: str


class StandardInfo(BaseModel):
    standard_id: str
    title: str
    category: str
    applications: List[str] = []
    keywords: List[str] = []


# --- Global state ---
rag_pipeline = None
standards_data = []
chunks_data = []
engine_mode = "initializing"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG pipeline on startup."""
    global rag_pipeline, standards_data, chunks_data, engine_mode

    logger.info("=" * 60)
    logger.info("  BIS Standards RAG Engine - Initializing...")
    logger.info("=" * 60)

    try:
        # 1. Load and chunk data
        data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "bis_standards.json")
        chunks_data = load_and_chunk_standards(data_path)

        with open(data_path, "r", encoding="utf-8") as f:
            standards_data = json.load(f)

        # 2. Build vector store
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        vector_store = BISVectorStore(persist_dir=persist_dir, model_name=model_name)
        vector_store.add_chunks(chunks_data)

        # 3. Build BM25 retriever
        bm25 = BM25Retriever(chunks_data)

        # 4. Create hybrid retriever
        dense_w = float(os.getenv("DENSE_WEIGHT", "0.6"))
        sparse_w = float(os.getenv("BM25_WEIGHT", "0.4"))
        hybrid = HybridRetriever(vector_store, bm25, dense_w, sparse_w)

        # 5. Load re-ranker
        reranker = ReRanker()

        # 6. Load LLM
        llm = _get_llm()
        engine_mode = "gemini" if llm else "fallback"

        # 7. Assemble pipeline
        rag_pipeline = RAGPipeline(hybrid, reranker, llm)

        logger.info("=" * 60)
        logger.info(f"  RAG Engine Ready! Mode: {engine_mode}")
        logger.info(f"  Standards: {len(standards_data)} | Chunks: {len(chunks_data)}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        engine_mode = "error"
        raise

    yield

    logger.info("Shutting down RAG engine...")


# --- FastAPI App ---

app = FastAPI(
    title="BIS Standards RAG Recommendation Engine",
    description=(
        "AI-powered engine that recommends relevant Bureau of Indian Standards (BIS) "
        "for Indian MSE products using RAG (Retrieval-Augmented Generation). "
        "Focused on Building Materials: Cement, Steel, Concrete, Aggregates."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# --- Routes ---

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the 3D frontend UI."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "BIS RAG Engine API v2.0. Visit /docs for API documentation."})


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with engine status."""
    return HealthResponse(
        status="healthy" if rag_pipeline else "degraded",
        engine="BIS RAG v2.0",
        standards_count=len(standards_data),
        chunks_count=len(chunks_data),
        llm_mode=engine_mode,
    )


@app.post("/api/recommend", response_model=RecommendationResponse)
async def recommend_standards(request: ProductRequest):
    """
    Recommend BIS standards for a given product description.
    
    Pipeline: Dense + BM25 Hybrid Retrieval -> Cross-Encoder Re-ranking -> LLM Rationale
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized yet. Please wait.")

    try:
        result = rag_pipeline.recommend(
            product_description=request.product_description,
            top_k=request.top_k,
            category_filter=request.category_filter,
        )
        return RecommendationResponse(**result)
    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.get("/api/standards", response_model=Dict[str, Any])
async def list_standards(category: Optional[str] = Query(None, description="Filter by category")):
    """List all BIS standards in the knowledge base."""
    filtered = standards_data
    if category:
        filtered = [s for s in standards_data if s["category"].lower() == category.lower()]

    return {
        "total": len(filtered),
        "categories": list(set(s["category"] for s in standards_data)),
        "standards": [
            StandardInfo(
                standard_id=s["standard_id"],
                title=s["title"],
                category=s["category"],
                applications=s.get("applications", []),
                keywords=s.get("keywords", []),
            ).model_dump()
            for s in filtered
        ],
    }


@app.get("/api/standard/{standard_id}")
async def get_standard(standard_id: str):
    """Get detailed information about a specific BIS standard."""
    # Handle URL encoding (e.g., "IS%20269:2015" -> "IS 269:2015")
    standard_id = standard_id.replace("%20", " ")
    
    for s in standards_data:
        if s["standard_id"] == standard_id:
            return s
    raise HTTPException(status_code=404, detail=f"Standard '{standard_id}' not found")


@app.get("/api/categories")
async def list_categories():
    """List all available categories with counts."""
    counts = {}
    for s in standards_data:
        cat = s["category"]
        counts[cat] = counts.get(cat, 0) + 1
    return {"categories": [{"name": k, "count": v} for k, v in sorted(counts.items())]}


@app.get("/api/search")
async def quick_search(q: str = Query(..., min_length=2, description="Search query")):
    """Quick search across standard titles and keywords."""
    q_lower = q.lower()
    matches = []
    for s in standards_data:
        score = 0
        if q_lower in s["title"].lower():
            score += 10
        if q_lower in s["standard_id"].lower():
            score += 20
        for kw in s.get("keywords", []):
            if q_lower in kw.lower():
                score += 5
        if q_lower in s.get("scope", "").lower():
            score += 2
        if score > 0:
            matches.append({"standard_id": s["standard_id"], "title": s["title"], "category": s["category"], "score": score})

    matches.sort(key=lambda x: x["score"], reverse=True)
    return {"query": q, "results": matches[:10]}


# Keep backward compatibility with old endpoint
@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_standards_legacy(request: ProductRequest):
    """Legacy endpoint - redirects to /api/recommend."""
    return await recommend_standards(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
