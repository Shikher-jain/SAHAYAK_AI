from __future__ import annotations

from typing import Dict, List, Sequence

from backend.services import vector_service


def recommend(items: Sequence[Dict[str, str]], query_embedding=None, top_k: int = 3) -> List[Dict[str, str]]:
    """Simple deterministic fallback for callers that pass pre-ranked items."""
    return list(items[:top_k])


class Recommender:
    """Recommendation helper backed by the unified vector service."""

    def __init__(self, target: str = "auto") -> None:
        self.target = target

    def recommend(self, query_text: str, top_k: int = 5) -> List[Dict[str, str]]:
        if not query_text:
            return []
        return vector_service.search_vectors(query_text, top_k=top_k, target=self.target)
