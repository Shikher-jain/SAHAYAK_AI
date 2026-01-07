from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from transformers import pipeline

from backend.ingestion.text import chunk_text
from backend.local_stack import db as local_db
from backend.local_stack import embedder as local_embedder
from backend.vector_store import qdrant_store

local_db.init_db()

logger = logging.getLogger("sahayak.vector_service")

_summary_pipeline = None


def _load_summarizer():
    global _summary_pipeline
    if _summary_pipeline is None:
        try:
            _summary_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception:
            _summary_pipeline = None
    return _summary_pipeline


def _use_qdrant(target: str) -> bool:
    if target == "qdrant":
        if not qdrant_store.is_available:
            raise RuntimeError("Qdrant backend is not available")
        return True
    if target == "auto":
        return qdrant_store.is_available
    return False


def _use_local(target: str) -> bool:
    if target == "local":
        return True
    if target == "auto" and not qdrant_store.is_available:
        return True
    return target == "auto"


def ingest_text(text: str, metadata: Dict[str, str] | None = None, target: str = "auto") -> List[Dict[str, str]]:
    metadata = metadata or {}
    segments = chunk_text(text) or [text]
    records: List[Dict[str, str]] = []
    for segment in segments:
        embedding = local_embedder.embed_text(segment)
        if _use_qdrant(target):
            try:
                record = qdrant_store.upsert_text(segment, metadata, embedding)
                record["backend"] = "qdrant"
                records.append(record)
            except Exception as exc:
                logger.warning("Qdrant ingestion failed, falling back to local store: %s", exc)
        if _use_local(target):
            records.append(_ingest_local(segment, metadata, embedding))
    return records


def _ingest_local(text: str, metadata: Dict[str, str], embedding: np.ndarray | None = None) -> Dict[str, str]:
    embedding = embedding if embedding is not None else local_embedder.embed_text(text)
    filename = metadata.get("source", "local-upload")
    local_db.add_chunk(filename, text, embedding)
    return {"backend": "local", "metadata": metadata, "content": text}


def search_vectors(query: str, top_k: int = 5, target: str = "auto") -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    query_embedding = local_embedder.embed_text(query)
    if _use_qdrant(target):
        try:
            results.extend(_search_qdrant(query_embedding, top_k))
        except Exception as exc:
            logger.warning("Qdrant search failed, falling back to local store: %s", exc)
    if _use_local(target):
        results.extend(_search_local(query_embedding, top_k))
    # Deduplicate by id while keeping highest score
    deduped: Dict[str, Dict[str, str]] = {}
    for item in results:
        key = item.get("id") or f"local-{len(deduped)}"
        if key not in deduped or item.get("score", 0) > deduped[key].get("score", 0):
            deduped[key] = item
    sorted_hits = sorted(deduped.values(), key=lambda r: r.get("score", 0), reverse=True)
    return sorted_hits[:top_k]


def _search_qdrant(query_embedding: np.ndarray, top_k: int) -> List[Dict[str, str]]:
    hits = qdrant_store.search(query_embedding, top_k)
    for hit in hits:
        hit.setdefault("backend", "qdrant")
    return hits


def _search_local(query_embedding: np.ndarray, top_k: int) -> List[Dict[str, str]]:
    index, texts = local_db.build_faiss_index()
    if index.ntotal == 0:
        return []
    query_vec = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query_vec, top_k)
    hits: List[Dict[str, str]] = []
    for distance, idx in zip(distances[0], indices[0]):
        idx_int = int(idx)
        if idx_int >= len(texts):
            continue
        score = float(1 / (1 + distance))
        hits.append({
            "id": f"local-{idx_int}",
            "score": score,
            "metadata": {"source": "local", "chunk": idx_int},
            "content": texts[idx_int],
        })
    return hits


def rag_answer(query: str, top_k: int = 5, target: str = "auto") -> Dict[str, str]:
    hits = search_vectors(query, top_k=top_k, target=target)
    context = "\n\n".join(hit.get("content", "") for hit in hits if hit.get("content"))
    if not context:
        return {"answer": "No context available yet. Please ingest content first.", "sources": []}
    synthesized = summarize_text(f"{context}\n\nQuestion: {query}")
    return {"answer": synthesized, "context": context, "sources": hits}


def summarize_text(text: str, max_length: int = 160) -> str:
    summarizer = _load_summarizer()
    snippet = text.strip()
    if not snippet:
        return ""
    if summarizer:
        result = summarizer(snippet[:1024], max_length=max_length, min_length=60, do_sample=False)
        return result[0]["summary_text"]
    # Fallback: return first sentences
    sentences = snippet.split(".")
    return ".".join(sentences[:3]).strip()
