"""
Sahayak AI — Query Rewrite & Expansion.

Two-tier strategy:
1. Fast path: lightweight rule-based normalization (always runs).
2. LLM path: optional intelligent expansion using Groq/OpenAI to add
   synonyms, related concepts, and clarifying terms.

Falls back gracefully when no LLM is available.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.common.embedder import embed_text, embed_texts
from dotenv import load_dotenv
load_dotenv()

def rewrite_query(query: str) -> str:
    """
    Rewrite the user query before embedding.

    Applies rule-based normalization first, then attempts LLM expansion
    if a backend is available.  Returns the expanded query string.
    """
    cleaned = _normalize(query)
    expanded = _llm_expand(cleaned)
    return expanded or cleaned


# ------------------------------------------------------------------
# Rule-based normalization
# ------------------------------------------------------------------

def _normalize(query: str) -> str:
    """Basic normalization: strip, collapse whitespace, remove trailing punctuation."""
    q = query.strip()
    # Collapse multiple spaces
    while "  " in q:
        q = q.replace("  ", " ")
    # Remove trailing question mark (helps embedding models focus on content)
    if q.endswith("?"):
        q = q[:-1].strip()
    return q


# ------------------------------------------------------------------
# LLM-based query expansion
# ------------------------------------------------------------------

def _llm_expand(query: str) -> Optional[str]:
    """
    Use a lightweight LLM call to expand the query with related terms.

    The expansion is appended to the original query so the embedding
    captures broader semantic context.
    """
    if os.getenv("GROQ_API_KEY"):
        return _expand_groq(query)
    if os.getenv("OPENAI_API_KEY"):
        return _expand_openai(query)
    return None


def _expand_groq(query: str) -> Optional[str]:
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query expansion engine. Given a user question, "
                        "output 3-5 key search terms or related concepts that would help "
                        "find relevant documents. Output ONLY the terms, comma-separated. "
                        "Do not answer the question."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.2,
            max_tokens=60,
        )
        expansion = (resp.choices[0].message.content or "").strip()
        return _merge_expansion(query, expansion)
    except Exception:
        return None


def _expand_openai(query: str) -> Optional[str]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query expansion engine. Given a user question, "
                        "output 3-5 key search terms or related concepts that would help "
                        "find relevant documents. Output ONLY the terms, comma-separated. "
                        "Do not answer the question."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.2,
            max_tokens=60,
        )
        expansion = (resp.choices[0].message.content or "").strip()
        return _merge_expansion(query, expansion)
    except Exception:
        return None


def _merge_expansion(query: str, expansion: str) -> Optional[str]:
    """Merge expansion terms into the query, deduplicating."""
    if not expansion:
        return None
    # Parse comma-separated terms
    terms = [t.strip().lower() for t in expansion.replace(";", ",").split(",") if t.strip()]
    query_lower = query.lower()
    # Only keep terms not already in the query
    new_terms = [t for t in terms if t and t not in query_lower and len(t) > 2]
    if not new_terms:
        return None
    # Limit to 5 expansion terms
    return f"{query} {' '.join(new_terms[:5])}"


# ------------------------------------------------------------------
# Semantic query expansion (uses embeddings — no LLM needed)
# ------------------------------------------------------------------

class QueryRewriter:
    def __init__(self):
        pass

    def expand_query(self, query: str, related_phrases: Optional[List[str]] = None, top_k: int = 3) -> str:
        """
        Expand query by selecting the most semantically similar phrases
        from a candidate list.  Uses embedding cosine similarity.
        """
        if not related_phrases or len(related_phrases) == 0:
            return query

        query_vec = embed_text(query)
        phrases_vec = embed_texts(related_phrases)

        sims = cosine_similarity([query_vec], phrases_vec)[0]
        top_indices = sims.argsort()[::-1][:top_k]
        top_phrases = [related_phrases[i] for i in top_indices]

        expanded_query = query + " " + " ".join(top_phrases)
        return expanded_query
