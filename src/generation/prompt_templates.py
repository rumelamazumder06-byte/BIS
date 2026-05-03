"""
BIS Standards RAG Engine - Prompt Templates
Structured prompts for the LLM to generate BIS standard recommendations.
"""

SYSTEM_PROMPT = """You are an expert consultant on Bureau of Indian Standards (BIS) regulations, 
specializing in Building Materials (Cement, Steel, Concrete, and Aggregates).

Your role is to analyze a product description provided by an Indian Micro or Small Enterprise (MSE) 
and recommend the most relevant BIS standards they must comply with.

RULES:
1. Only recommend standards from the provided context excerpts.
2. Provide a clear, concise rationale for each recommendation.
3. Rank standards by relevance (most relevant first).
4. Assign a confidence level: High, Medium, or Low.
5. Mention specific clauses when relevant.
6. Output valid JSON only — no markdown, no extra text.

OUTPUT FORMAT (strict JSON array):
[
  {
    "rank": 1,
    "standard_id": "IS XXXX:YYYY",
    "title": "Full title of the standard",
    "rationale": "2-3 sentence explanation of why this standard applies to the product",
    "confidence": "High",
    "key_clauses": ["Clause X: brief description"]
  }
]
"""

USER_PROMPT_TEMPLATE = """
PRODUCT DESCRIPTION:
{product_description}

RELEVANT BIS STANDARD EXCERPTS:
{context}

Based on the above excerpts, recommend the top {top_k} most relevant BIS standards for this product.
Return ONLY a JSON array. Do not include any other text.
"""


def build_context_from_docs(documents: list) -> str:
    """Build context string from retrieved documents."""
    context_parts = []
    for i, doc in enumerate(documents, 1):
        std_id = doc["metadata"]["standard_id"]
        title = doc["metadata"]["title"]
        category = doc["metadata"]["category"]
        text = doc["text"]
        context_parts.append(
            f"--- Excerpt {i} ---\n"
            f"Standard: {std_id} — {title}\n"
            f"Category: {category}\n"
            f"Content: {text}\n"
        )
    return "\n".join(context_parts)


def build_rag_prompt(product_description: str, documents: list, top_k: int = 5) -> tuple:
    """Build the full RAG prompt (system + user) for the LLM."""
    context = build_context_from_docs(documents)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        product_description=product_description,
        context=context,
        top_k=top_k,
    )
    return SYSTEM_PROMPT, user_prompt
