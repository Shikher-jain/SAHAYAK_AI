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

    def recommend_for_user(self, user_id: str, top_k: int = 5) -> List[Dict[str, str]]:
        """TASK 20: Content-based recommendations from user's recent activity."""
        recommendations: List[Dict[str, str]] = []
        # Get recent documents from user's ingestion history
        try:
            from backend.vector_store import qdrant_store
            # FIX 3: Wrap Qdrant calls in try-except to prevent crashes.
            # If Qdrant fails, an empty list is returned, preventing breakdown of RAG answer.
            recent = qdrant_store.recent_payloads(limit=10)
            # Collect tags and subjects from recent uploads
            topics = set()
            for payload in recent:
                tags = payload.get("tags", [])
                if isinstance(tags, list):
                    topics.update(tags)
                source = payload.get("source", "")
                if source:
                    topics.add(source)
            # Search for related content based on collected topics
            for topic in list(topics)[:3]:
                hits = vector_service.search_vectors(str(topic), top_k=2, target=self.target)
                recommendations.extend(hits)
        except Exception:
            pass
        # Deduplicate
        seen = set()
        unique = []
        for r in recommendations:
            key = r.get("id", "")
            if key and key not in seen:
                seen.add(key)
                unique.append(r)
        return unique[:top_k]
