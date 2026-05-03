"""
BIS Standards RAG Engine - Streamlit Demo UI
Beautiful, interactive UI for querying BIS standard recommendations.
"""

import os
import sys
import json
import time

import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
load_dotenv()

from src.ingestion.chunker import load_and_chunk_standards
from src.ingestion.embedder import BISVectorStore
from src.retrieval.hybrid_retriever import BM25Retriever, HybridRetriever
from src.retrieval.reranker import ReRanker
from src.generation.llm_chain import RAGPipeline, GeminiLLM


# --- Page Config ---
st.set_page_config(
    page_title="BIS Standards Finder",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3a5f, #2d8cf0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .rec-card {
        background: linear-gradient(135deg, #f8f9ff, #e8eeff);
        border-left: 4px solid #2d8cf0;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .confidence-high { color: #0f9d58; font-weight: 700; }
    .confidence-medium { color: #f4b400; font-weight: 700; }
    .confidence-low { color: #db4437; font-weight: 700; }
    .metric-box {
        background: #1e3a5f;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.8; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_rag_pipeline():
    """Initialize the RAG pipeline (cached across reruns)."""
    with st.spinner("🔧 Loading RAG engine... (first time takes ~30s)"):
        # Load data
        data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "bis_standards.json")
        chunks = load_and_chunk_standards(data_path)

        # Vector store
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        vector_store = BISVectorStore(persist_dir=persist_dir, model_name=model_name)
        vector_store.add_chunks(chunks)

        # BM25
        bm25 = BM25Retriever(chunks)

        # Hybrid retriever
        hybrid = HybridRetriever(vector_store, bm25)

        # Re-ranker
        reranker = ReRanker()

        # LLM (optional)
        llm = None
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if api_key and api_key != "your_gemini_api_key_here":
            try:
                llm = GeminiLLM(api_key=api_key)
            except Exception:
                pass

        return RAGPipeline(hybrid, reranker, llm)


# --- Initialize ---
pipeline = init_rag_pipeline()

# --- Header ---
st.markdown('<h1 class="main-header">🏗️ BIS Standards Finder</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered recommendation engine for Bureau of Indian Standards — Building Materials</p>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of recommendations", 1, 10, 5)
    category = st.selectbox("Category filter", ["All", "Cement", "Steel", "Concrete", "Aggregates"])
    category_filter = None if category == "All" else category

    st.divider()
    st.header("📝 Example Queries")
    examples = [
        "53 grade ordinary portland cement for high-rise building construction",
        "High strength TMT steel bars Fe 500D grade for RCC bridge construction",
        "Ready-mix concrete M30 grade with superplasticizer for commercial building",
        "Crushed stone coarse aggregate 20mm nominal size for concrete production",
        "Portland slag cement for mass concrete dam construction in marine environment",
        "Steel reinforcement bars for residential building foundation and columns",
    ]
    for ex in examples:
        if st.button(f"📌 {ex[:60]}...", key=ex, use_container_width=True):
            st.session_state["query_input"] = ex

    st.divider()
    st.caption("Built for BIS Hackathon 2026 🇮🇳")
    st.caption("RAG Pipeline: Dense + BM25 + Cross-Encoder")

# --- Main Input ---
query = st.text_area(
    "🔍 Describe your product:",
    value=st.session_state.get("query_input", ""),
    height=100,
    placeholder="e.g., 53 grade ordinary portland cement for structural concrete in high-rise buildings...",
    key="main_query",
)

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    search_clicked = st.button("🚀 Find Standards", type="primary", use_container_width=True)

# --- Results ---
if search_clicked and query.strip():
    with st.spinner("🔎 Searching BIS standards database..."):
        result = pipeline.recommend(
            product_description=query,
            top_k=top_k,
            category_filter=category_filter,
        )

    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-value">{len(result['recommendations'])}</div>
            <div class="metric-label">Standards Found</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-value">{result['latency_ms']:.0f}ms</div>
            <div class="metric-label">Response Time</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-value">{result['method']}</div>
            <div class="metric-label">Engine Mode</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Recommendation cards
    for rec in result["recommendations"]:
        conf_class = f"confidence-{rec['confidence'].lower()}"

        st.markdown(f"""
        <div class="rec-card">
            <h3>#{rec['rank']} — {rec['standard_id']}</h3>
            <h4>{rec['title']}</h4>
            <p><strong>Confidence:</strong> <span class="{conf_class}">{rec['confidence']}</span></p>
            <p><strong>Rationale:</strong> {rec['rationale']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Key clauses in expander
        if rec.get("key_clauses"):
            with st.expander(f"📋 Key Clauses — {rec['standard_id']}"):
                for clause in rec["key_clauses"]:
                    st.markdown(f"- {clause}")

    # Raw JSON output
    with st.expander("📦 Raw JSON Response"):
        st.json(result)

elif search_clicked:
    st.warning("Please enter a product description.")
