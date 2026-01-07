from __future__ import annotations

from typing import Dict, List

from backend.services import vector_service


def retrieve(query_text: str, top_k: int = 5, target: str = "auto") -> List[Dict[str, str]]:
    """Return semantic search hits using the active vector backends."""
    if not query_text:
        return []
    return vector_service.search_vectors(query_text, top_k=top_k, target=target)


class Retriever:
    """Backward-compatible helper that mirrors the legacy interface."""

    def __init__(self, target: str = "auto") -> None:
        self.target = target

    def search_vectors(self, query_text: str, top_k: int = 5) -> List[Dict[str, str]]:
        return retrieve(query_text, top_k=top_k, target=self.target)
