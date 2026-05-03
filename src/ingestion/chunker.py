"""
BIS Standards RAG Engine - Chunking Module
Converts BIS standard records into retrievable document chunks with metadata.
"""

import json
import os
from typing import List, Dict


def create_chunks_from_standard(standard: Dict) -> List[Dict]:
    """
    Create hierarchical chunks from a single BIS standard record.
    
    Level 0: Full scope + title (broad retrieval)
    Level 1: Individual clauses (precise retrieval)
    Level 2: Application-focused chunk (use-case matching)
    """
    chunks = []
    std_id = standard["standard_id"]
    title = standard["title"]
    category = standard["category"]
    keywords = standard.get("keywords", [])

    # --- Level 0: Scope chunk (always retrieved for broad matching) ---
    scope_text = (
        f"BIS Standard: {std_id} — {title}\n"
        f"Category: {category}\n"
        f"Scope: {standard['scope']}\n"
        f"Applications: {', '.join(standard.get('applications', []))}\n"
        f"Keywords: {', '.join(keywords)}"
    )
    chunks.append({
        "chunk_id": f"{std_id}_scope",
        "standard_id": std_id,
        "title": title,
        "category": category,
        "level": 0,
        "text": scope_text,
        "keywords": keywords,
    })

    # --- Level 1: Individual clause chunks ---
    for i, clause in enumerate(standard.get("key_clauses", [])):
        clause_text = (
            f"BIS Standard: {std_id} — {title}\n"
            f"Category: {category}\n"
            f"{clause}"
        )
        chunks.append({
            "chunk_id": f"{std_id}_clause_{i}",
            "standard_id": std_id,
            "title": title,
            "category": category,
            "level": 1,
            "text": clause_text,
            "keywords": keywords,
        })

    # --- Level 2: Application-focused chunk ---
    applications = standard.get("applications", [])
    if applications:
        app_text = (
            f"BIS Standard: {std_id} — {title}\n"
            f"Category: {category}\n"
            f"This standard is applicable for: {', '.join(applications)}.\n"
            f"Scope summary: {standard['scope'][:200]}"
        )
        chunks.append({
            "chunk_id": f"{std_id}_applications",
            "standard_id": std_id,
            "title": title,
            "category": category,
            "level": 2,
            "text": app_text,
            "keywords": keywords,
        })

    return chunks


def load_and_chunk_standards(data_path: str) -> List[Dict]:
    """Load BIS standards JSON and create all chunks."""
    with open(data_path, "r", encoding="utf-8") as f:
        standards = json.load(f)

    all_chunks = []
    for standard in standards:
        chunks = create_chunks_from_standard(standard)
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks from {len(standards)} standards")
    return all_chunks


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "bis_standards.json")
    chunks = load_and_chunk_standards(data_path)
    for c in chunks[:3]:
        print(f"\n--- {c['chunk_id']} (level {c['level']}) ---")
        print(c["text"][:200])
