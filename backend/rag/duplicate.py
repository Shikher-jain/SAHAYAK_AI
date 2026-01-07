from __future__ import annotations

import hashlib
from typing import Dict, List, Set

import numpy as np

from backend.local_stack import embedder as local_embedder
from backend.services import vector_service


def is_duplicate(text: str, existing_hashes: Set[str]) -> bool:
    """Check whether the md5 hash of *text* already exists."""
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    return text_hash in existing_hashes


class DuplicateDetector:
    """Embeddings-based duplicate detector that works with any backend."""

    def __init__(self, threshold: float = 0.85, target: str = "auto") -> None:
        self.threshold = threshold
        self.target = target

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def check_duplicates(self, new_text: str, top_k: int = 5) -> List[Dict[str, str]]:
        if not new_text.strip():
            return []
        query_vec = local_embedder.embed_text(new_text)
        candidates = vector_service.search_vectors(new_text, top_k=top_k, target=self.target)
        duplicates: List[Dict[str, str]] = []
        for candidate in candidates:
            content = candidate.get("content", "")
            if not content:
                continue
            candidate_vec = local_embedder.embed_text(content)
            similarity = self._cosine(query_vec, candidate_vec)
            if similarity >= self.threshold:
                duplicates.append({
                    "text": content,
                    "metadata": candidate.get("metadata", {}),
                    "similarity": similarity,
                })
        return duplicates
