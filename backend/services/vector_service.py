from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List

import numpy as np
from transformers import pipeline

from backend.ingestion.text import chunk_text
from backend.local_stack import db as local_db
from backend.local_stack import embedder as local_embedder
from backend.vector_store import qdrant_store

local_db.init_db()

logger = logging.getLogger("sahayak.vector_service")

_summary_pipeline = None

_SANITIZE_TRANSLATION = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
})


def _sanitize_output(text: str) -> str:
    """Remove emojis/non-ASCII glyphs and normalize whitespace for API responses."""
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text).translate(_SANITIZE_TRANSLATION)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.replace("\r\n", "\n").replace("\r", "\n")
    ascii_text = re.sub(r"[ \t]+", " ", ascii_text)
    ascii_text = re.sub(r"\n{3,}", "\n\n", ascii_text)

    cleaned_lines = []
    previous_blank = False
    for raw_line in ascii_text.split("\n"):
        stripped_line = raw_line.strip()
        if not stripped_line:
            if cleaned_lines and not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue
        cleaned_lines.append(stripped_line)
        previous_blank = False

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned


def _sanitize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(record)
    content = sanitized.get("content")
    if isinstance(content, str):
        sanitized["content"] = _sanitize_output(content)

    metadata = sanitized.get("metadata")
    if isinstance(metadata, dict):
        cleaned_meta = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                cleaned_meta[key] = _sanitize_output(value)
            else:
                cleaned_meta[key] = value
        sanitized["metadata"] = cleaned_meta
    return sanitized


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
    sanitized_hits = [_sanitize_record(hit) for hit in sorted_hits[:top_k]]
    return sanitized_hits


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
    sanitized_context = _sanitize_output(context)
    sanitized_query = _sanitize_output(query)
    if not sanitized_context:
        return {"answer": "No context available yet. Please ingest content first.", "sources": []}
    synthesized = summarize_text(f"{sanitized_context}\n\nQuestion: {sanitized_query}")
    return {"answer": synthesized, "context": sanitized_context, "sources": hits}


def summarize_text(text: str, max_length: int = 160) -> str:
    summarizer = _load_summarizer()
    snippet = text.strip()
    if not snippet:
        return ""
    if summarizer:
        result = summarizer(snippet[:1024], max_length=max_length, min_length=60, do_sample=False)
        raw_summary = result[0]["summary_text"]
    # Fallback: return first sentences
    else:
        sentences = snippet.split(".")
        raw_summary = ".".join(sentences[:3]).strip()
    return _sanitize_output(raw_summary)
