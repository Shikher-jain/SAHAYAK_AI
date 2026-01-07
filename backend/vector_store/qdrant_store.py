from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except ImportError:  # pragma: no cover - optional dependency
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore


logger = logging.getLogger("sahayak.qdrant")


class QdrantStore:
    def __init__(self) -> None:
        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION", "sahayak_ai_vectors")
        self.vector_dim = int(os.getenv("QDRANT_VECTOR_DIM", "384"))
        self._client: QdrantClient | None = None
        self._available = False
        if QdrantClient is None:
            logger.warning("qdrant-client is not installed; remote vector store disabled.")
            return
        self._connect()

    @property
    def is_available(self) -> bool:
        return self._available and self._client is not None

    def status(self) -> Dict[str, Any]:
        return {
            "available": self.is_available,
            "url": self.url,
            "collection": self.collection_name,
            "vector_dim": self.vector_dim,
        }

    def _connect(self) -> None:
        try:
            self._client = QdrantClient(url=self.url, api_key=self.api_key or None, timeout=5.0)
            self._available = True
            self._ensure_collection()
        except Exception as exc:  # pragma: no cover - connectivity
            logger.warning("Unable to reach Qdrant at %s: %s", self.url, exc)
            self._client = None
            self._available = False

    def _ensure_collection(self) -> None:
        if not self._client:
            return
        try:
            self._client.get_collection(self.collection_name)
        except Exception:
            vectors_config = qmodels.VectorParams(size=self.vector_dim, distance=qmodels.Distance.COSINE)
            self._client.recreate_collection(collection_name=self.collection_name, vectors_config=vectors_config)

    def upsert_text(self, text: str, metadata: Dict[str, Any], embedding: np.ndarray) -> Dict[str, Any]:
        if not self._client:
            raise RuntimeError("Qdrant client is not available")
        vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        payload = {**metadata, "content": text}
        point_id = uuid.uuid4().hex
        point = qmodels.PointStruct(id=point_id, vector=vector, payload=payload)
        self._client.upsert(collection_name=self.collection_name, points=[point])
        return {"id": point_id, "metadata": metadata, "content": text}

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._client:
            raise RuntimeError("Qdrant client is not available")
        vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
        hits: List[Dict[str, Any]] = []
        for hit in results:
            payload = hit.payload or {}
            hits.append(
                {
                    "id": str(hit.id),
                    "score": float(hit.score or 0.0),
                    "metadata": payload,
                    "content": payload.get("content", ""),
                }
            )
        return hits

    def recent_payloads(self, limit: int = 10) -> List[Dict[str, Any]]:
        if not self._client:
            return []
        try:
            points, _ = self._client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
            )
            return [point.payload or {} for point in points]
        except Exception:
            return []


def _build_store() -> QdrantStore:
    return QdrantStore()


qdrant_store = _build_store()
