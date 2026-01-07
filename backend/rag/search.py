from __future__ import annotations

from typing import Dict, List

from backend.rag.retriever import retrieve
from backend.services import vector_service


def semantic_search(query_text: str, top_k: int = 5, target: str = "auto") -> List[Dict[str, str]]:
    """Expose a minimal semantic-search helper for legacy imports."""
    return retrieve(query_text, top_k=top_k, target=target)


class RAGSearcher:
    """Small convenience wrapper around the core vector service."""

    def __init__(self, target: str = "auto", top_k: int = 5) -> None:
        self.target = target
        self.top_k = top_k

    def query(self, query_text: str | None, top_k: int | None = None) -> List[Dict[str, str]]:
        if not query_text:
            return []
        return semantic_search(query_text, top_k=top_k or self.top_k, target=self.target)

    def answer(self, question: str, top_k: int | None = None) -> Dict[str, str]:
        return vector_service.rag_answer(question, top_k=top_k or self.top_k, target=self.target)
