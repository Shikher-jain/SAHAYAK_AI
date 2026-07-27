from __future__ import annotations

from typing import Dict, List

from backend.services import vector_service

# Minimum cosine-similarity score for a retrieved chunk to be considered
# "relevant" (as opposed to just "closest of a bad bunch"). Without this,
# vector search always returns its top-k nearest neighbors even when none
# of them are actually related to the query — e.g. asking about "Pokemon"
# against a technical PDF still returns the PDF's least-irrelevant chunks,
# which then get treated as real context and can trigger hallucination
# instead of the "Out of Context" response.
#
# Tune this if needed: too high -> valid answers get dropped as "no context
# found"; too low -> irrelevant chunks slip through again. 0.3 is a
# reasonable starting point for normalized sentence-transformer embeddings
# with cosine distance (matches all-MiniLM-L6-v2, the default embedder).
MIN_RELEVANCE_SCORE = 0.3


def retrieve(query_text: str, top_k: int = 5, target: str = "auto") -> List[Dict[str, str]]:
    """Return semantic search hits using the active vector backends,
    filtered to only those above MIN_RELEVANCE_SCORE. If nothing clears
    the bar, returns an empty list — which is what triggers the "Out of
    Context" path in system_prompt.build_user_prompt(), instead of feeding
    the LLM irrelevant chunks and letting it guess."""
    if not query_text:
        return []
    hits = vector_service.search_vectors(query_text, top_k=top_k, target=target)
    return [h for h in hits if h.get("score", 0) >= MIN_RELEVANCE_SCORE]


class Retriever:
    """Backward-compatible helper that mirrors the legacy interface."""

    def __init__(self, target: str = "auto") -> None:
        self.target = target

    def search_vectors(self, query_text: str, top_k: int = 5) -> List[Dict[str, str]]:
        return retrieve(query_text, top_k=top_k, target=self.target)