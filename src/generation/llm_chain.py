"""
BIS Standards RAG Engine - LLM Chain
Orchestrates the full RAG pipeline: Retrieve → Re-rank → Generate.
Uses google.genai (new SDK) instead of deprecated google.generativeai.
"""

import json
import time
import os
import re
from typing import List, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

from src.generation.prompt_templates import build_rag_prompt, SYSTEM_PROMPT


def _get_llm():
    """Factory to create the appropriate LLM based on available API key."""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if api_key and api_key != "your_gemini_api_key_here":
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            return GeminiLLM(client=client)
        except ImportError:
            print("google-genai not installed. Trying legacy package...")
            try:
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=api_key)
                return GeminiLegacyLLM(genai_legacy)
            except ImportError:
                print("No Gemini SDK available. Using fallback.")
    return None


class GeminiLLM:
    """Wrapper for Google Gemini API using the new google.genai SDK."""

    def __init__(self, client=None, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        if client:
            self.client = client
        else:
            from google import genai
            api_key = os.getenv("GOOGLE_API_KEY", "")
            self.client = genai.Client(api_key=api_key)
        print(f"Gemini LLM ready: {model_name} (new SDK)")

    def generate(self, user_prompt: str) -> str:
        """Generate a response from the LLM."""
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=2048,
            ),
        )
        return response.text


class GeminiLegacyLLM:
    """Fallback wrapper using deprecated google.generativeai."""

    def __init__(self, genai_module, model_name: str = "gemini-1.5-flash"):
        self.model = genai_module.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai_module.GenerationConfig(
                temperature=0.1,
                max_output_tokens=2048,
            ),
        )
        print(f"Gemini LLM ready: {model_name} (legacy SDK)")

    def generate(self, user_prompt: str) -> str:
        response = self.model.generate_content(user_prompt)
        return response.text


class FallbackLLM:
    """
    Fallback LLM that generates recommendations without an external API.
    Uses the retrieved documents directly to produce structured output.
    """

    def generate_from_docs(self, product_description: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """Generate recommendations directly from re-ranked documents."""
        seen_standards = set()
        recommendations = []

        for doc in documents:
            std_id = doc["metadata"]["standard_id"]
            if std_id in seen_standards:
                continue
            seen_standards.add(std_id)

            score = doc.get("rerank_score", doc.get("fused_score", doc.get("score", 0)))
            if score > 0.7:
                confidence = "High"
            elif score > 0.4:
                confidence = "Medium"
            else:
                confidence = "Low"

            text = doc["text"]
            clauses = []
            for line in text.split("\n"):
                if line.strip().startswith("Clause"):
                    clauses.append(line.strip()[:120])

            # Build a better rationale from the document text
            scope_match = re.search(r"Scope:\s*(.+?)(?:\n|$)", text)
            scope_snippet = scope_match.group(1)[:150] if scope_match else ""

            rationale = (
                f"This standard covers {doc['metadata']['category'].lower()} specifications "
                f"relevant to your product. {scope_snippet}"
            )

            recommendations.append({
                "rank": len(recommendations) + 1,
                "standard_id": std_id,
                "title": doc["metadata"]["title"],
                "rationale": rationale,
                "confidence": confidence,
                "key_clauses": clauses[:3] if clauses else ["See standard document for specific clauses"],
            })

            if len(recommendations) >= top_k:
                break

        return recommendations


class RAGPipeline:
    """
    Complete RAG Pipeline: Retrieve -> Re-rank -> Generate.
    Main entry point for the recommendation engine.
    """

    def __init__(self, hybrid_retriever, reranker, llm=None):
        self.retriever = hybrid_retriever
        self.reranker = reranker
        self.llm = llm
        self.fallback = FallbackLLM()

    def recommend(
        self,
        product_description: str,
        top_k: int = 5,
        retrieval_k: int = 20,
        category_filter: Optional[str] = None,
    ) -> Dict:
        """Full RAG pipeline execution."""
        start_time = time.time()

        # Step 1: Hybrid Retrieval
        retrieved_docs = self.retriever.retrieve(
            query=product_description,
            top_k=retrieval_k,
            category_filter=category_filter,
        )

        # Step 2: Re-rank
        reranked_docs = self.reranker.rerank(
            query=product_description,
            documents=retrieved_docs,
            top_k=top_k * 2,
        )

        # Step 3: Generate recommendations
        method = "fallback-rerank"
        recommendations = []
        try:
            if self.llm:
                system_prompt, user_prompt = build_rag_prompt(
                    product_description, reranked_docs[:top_k * 2], top_k
                )
                raw_response = self.llm.generate(user_prompt)
                # Parse JSON from response (handle markdown code blocks)
                raw_response = raw_response.strip()
                if raw_response.startswith("```"):
                    raw_response = raw_response.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                recommendations = json.loads(raw_response)
                method = "gemini-rag"
            else:
                raise ValueError("No LLM configured")
        except Exception as e:
            print(f"LLM generation failed ({e}), using fallback...")
            recommendations = self.fallback.generate_from_docs(
                product_description, reranked_docs, top_k
            )
            method = "fallback-rerank"

        latency_ms = (time.time() - start_time) * 1000

        # Build source documents summary
        source_docs = []
        for doc in reranked_docs[:top_k]:
            source_docs.append({
                "chunk_id": doc.get("chunk_id", ""),
                "standard_id": doc["metadata"]["standard_id"],
                "title": doc["metadata"]["title"],
                "category": doc["metadata"]["category"],
                "score": round(doc.get("rerank_score", doc.get("fused_score", doc.get("score", 0))), 4),
                "text_preview": doc["text"][:200],
            })

        return {
            "product_description": product_description,
            "recommendations": recommendations[:top_k],
            "latency_ms": round(latency_ms, 2),
            "source_chunks_count": len(retrieved_docs),
            "method": method,
            "source_documents": source_docs,
        }
