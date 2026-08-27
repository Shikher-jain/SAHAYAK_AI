from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv
load_dotenv()
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
        self.collection_name = os.getenv("QDRANT_COLLECTION", "sahayak_ai")
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
            # Only pass API key over HTTPS — Qdrant warns on insecure connections
            use_api_key = self.api_key if self.url.startswith("https") else None
            self._client = QdrantClient(url=self.url, api_key=use_api_key, timeout=5.0)
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

    def upsert_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> List[Dict[str, Any]]:
        if not self._client:
            raise RuntimeError("Qdrant client is not available")
        if not texts:
            return []
        vectors = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        ids = [uuid.uuid4().hex for _ in texts]
        payloads = []
        for text, metadata in zip(texts, metadatas):
            payloads.append({**metadata, "content": text})
        # TASK 2: upload_collection sends all vectors in one batch (vs per-chunk upsert loops).
        self._client.upload_collection(
            collection_name=self.collection_name,
            vectors=vectors,
            payload=payloads,
            ids=ids,
        )
        return [
            {"id": point_id, "metadata": metadata, "content": text}
            for point_id, metadata, text in zip(ids, metadatas, texts)
        ]

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors using the current Qdrant API (query_points)."""
        if not self._client:
            raise RuntimeError("Qdrant client is not available")
        vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        # qdrant-client 1.7+ replaced .search() with .query_points()
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        # QueryResponse.points is List[ScoredPoint] — each has .id, .score, .payload
        hits: List[Dict[str, Any]] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                {
                    "id": str(point.id),
                    "score": float(point.score or 0.0),
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

    def payloads_by_source(self, source: str, limit: int = 250) -> List[Dict[str, Any]]:
        """
        Fetch payloads for a specific source identifier.

        This is used to reconstruct full documents by `metadata["source"]` (filename/url/etc).
        Returns an empty list when Qdrant is unavailable or filtering is unsupported.
        """

        if not self._client or not qmodels:
            return []
        resolved = (source or "").strip()
        if not resolved:
            return []
        try:
            points, _ = self._client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="source",
                            match=qmodels.MatchValue(value=resolved),
                        )
                    ]
                ),
            )
            return [point.payload or {} for point in points]
        except Exception:
            return []


def _build_store() -> QdrantStore:
    return QdrantStore()


qdrant_store = _build_store()
