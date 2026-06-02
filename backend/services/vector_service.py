from __future__ import annotations

import logging
import re
import unicodedata
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, List, Tuple

import numpy as np

from backend.ingestion.text import chunk_text
from backend.local_stack import db as local_db
from backend.common.language import detect_language, translate_from_english, translate_to_english
from backend.common.embedder import embed_text, embed_texts
from backend.processing.summarization import summarize_text as processing_summarize_text
from backend.processing.tagging import Tagger
from backend.rag.conversation import ConversationManager
from backend.rag.duplicate import DuplicateDetector
from backend.rag.generator import Generator
from backend.rag.query_rewrite import rewrite_query
from backend.rag.recommend import Recommender
from backend.vector_store import qdrant_store
from backend.common.modes import resolve_mode, get_ui_hints

local_db.init_db()

logger = logging.getLogger("sahayak.vector_service")

_summary_pipeline = None
_qa_generator = Generator()
_conversation_manager = ConversationManager()

# TASK 24: LRU response cache for repeated RAG queries.
_RAG_CACHE_MAX = 128
_rag_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_rag_cache_lock = Lock()

_SANITIZE_TRANSLATION = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
})


def _sanitize_output(text: str) -> str:
    """Normalize whitespace and punctuation for API responses."""
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text).translate(_SANITIZE_TRANSLATION)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    cleaned_lines = []
    previous_blank = False
    for raw_line in normalized.split("\n"):
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
            from transformers import pipeline

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
    return False


def ingest_text(
    text: str,
    metadata: Dict[str, Any] | None = None,
    target: str = "auto",
    chunking_strategy: str = "recursive",
) -> List[Dict[str, str]]:
    metadata = metadata or {}
    detected_lang = detect_language(text)
    english_text = translate_to_english(text, detected_lang)
    if detected_lang != "en":
        metadata = dict(metadata)
        metadata["source_language"] = detected_lang
        metadata["translated_to_english"] = True
    # TASK 3 FIX: auto-select chunking strategy based on modality.
    modality = (metadata.get("modality") or "").lower()
    resolved_strategy = _resolve_chunking_strategy(modality, chunking_strategy)
    segments = chunk_text(english_text, strategy=resolved_strategy) or [english_text]
    records: List[Dict[str, str]] = []

    qdrant_mode = _use_qdrant(target)
    local_mode = _use_local(target)
    tagger = Tagger()
    duplicate_detector = DuplicateDetector(target=target)

    cleaned_segments: List[str] = []
    segment_metadatas: List[Dict[str, Any]] = []
    for segment in segments:
        cleaned_segment = segment.strip()
        if not cleaned_segment:
            continue
        # TASK 2 FIX: skip segments that are duplicates of existing content.
        if duplicate_detector.check_duplicates(cleaned_segment):
            continue
        tags = tagger.extract_keywords(cleaned_segment)
        # TASK 5 FIX: attach generated tags to metadata for vector storage.
        segment_metadata = dict(metadata)
        if tags:
            segment_metadata["tags"] = tags
        cleaned_segments.append(cleaned_segment)
        segment_metadatas.append(segment_metadata)

    if not cleaned_segments:
        return records

    # TASK 2: one embed_texts() call replaces N sequential encode() calls (~10–50x faster on large docs).
    embeddings = embed_texts(cleaned_segments)
    if qdrant_mode:
        try:
            # TASK 2: Qdrant upload_collection batches all points in a single client operation.
            records.extend(qdrant_store.upsert_texts(cleaned_segments, segment_metadatas, embeddings))
            for record in records:
                record["backend"] = "qdrant"
            if target == "qdrant" or not local_mode:
                return records
        except Exception as exc:
            logger.warning("Qdrant ingestion failed, falling back to local store: %s", exc)
            if target == "qdrant":
                raise
            local_mode = True
    if local_mode:
        for segment, segment_metadata, embedding in zip(cleaned_segments, segment_metadatas, embeddings):
            records.append(_ingest_local(segment, segment_metadata, embedding))
    return records


def _ingest_local(text: str, metadata: Dict[str, Any], embedding: np.ndarray | None = None) -> Dict[str, str]:
    embedding = embedding if embedding is not None else embed_text(text)
    filename = metadata.get("source", "local-upload")
    # BUG 2 FIX: persist metadata alongside local vectors for accurate retrieval.
    local_db.add_chunk_with_metadata(filename, text, embedding, metadata)
    return {"backend": "local", "metadata": metadata, "content": text}


def search_vectors(query: str, top_k: int = 5, target: str = "auto") -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    query_embedding = embed_text(query)
    # TASK 24: reuse cached search results for identical queries.
    cache_key = f"{query.strip().lower()}|{top_k}|{target}"
    # NOTE: search results are not cached because they may change after ingestion.
    qdrant_mode = _use_qdrant(target)
    local_mode = _use_local(target)

    if qdrant_mode:
        try:
            results.extend(_search_qdrant(query_embedding, top_k))
        except Exception as exc:
            logger.warning("Qdrant search failed, falling back to local store: %s", exc)
            if target == "qdrant":
                raise
            local_mode = True

    if local_mode:
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
    index, texts, metadatas = local_db.build_faiss_index_with_metadata()
    if index.ntotal == 0:
        return []
    query_vec = np.array([query_embedding], dtype="float32")
    scores, indices = index.search(query_vec, top_k)
    hits: List[Dict[str, str]] = []
    for score, idx in zip(scores[0], indices[0]):
        idx_int = int(idx)
        if idx_int >= len(texts):
            continue
        # BUG 3 FIX: inner product scores are cosine-aligned after normalization.
        score = float(score)
        metadata = metadatas[idx_int] if idx_int < len(metadatas) else {"source": "local"}
        hits.append({
            "id": f"local-{idx_int}",
            "score": score,
            "metadata": metadata,
            "content": texts[idx_int],
        })
    return hits


def _citations_from_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build citation dicts from retrieval hit metadata for the generator."""
    citations: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        meta = hit.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        citation = {
            "source": meta.get("source") or meta.get("filename") or meta.get("url"),
            "chunk_type": meta.get("chunk_type"),
            "page": meta.get("page"),
            "function_name": meta.get("function_name"),
            "class_name": meta.get("class_name"),
            "row_range": meta.get("row_range"),
        }
        citation = {key: value for key, value in citation.items() if value is not None and value != ""}
        if not citation:
            continue
        dedupe_key = "|".join(f"{key}={value}" for key, value in sorted(citation.items()))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        citations.append(citation)
    return citations


def rag_answer(
    query: str,
    top_k: int = 5,
    target: str = "auto",
    session_id: str | None = None,
    learning_mode: str = "student",
    user_mode: str | None = None,
) -> Dict[str, Any]:
    # Resolve user_mode from modes.py
    resolved_user_mode = resolve_mode(user_mode)
    # TASK 24: LRU cache lookup for repeated RAG queries (skip when session has history).
    cache_key = f"{query.strip().lower()}|{top_k}|{target}|{learning_mode}|{resolved_user_mode}" if not session_id else None
    if cache_key:
        with _rag_cache_lock:
            cached = _rag_cache.get(cache_key)
            if cached is not None:
                _rag_cache.move_to_end(cache_key)
                return {**cached, "cached": True}
    query_lang = detect_language(query)
    english_query = translate_to_english(query, query_lang)
    # TASK 1: rewrite query before embedding/retrieval.
    rewritten_query = rewrite_query(english_query)
    hits = search_vectors(rewritten_query, top_k=top_k, target=target)
    context = "\n\n".join(hit.get("content", "") for hit in hits if hit.get("content"))
    sanitized_context = _sanitize_output(context)
    sanitized_query = _sanitize_output(english_query)
    if not sanitized_context:
        return {
            "answer": translate_from_english("No context available yet. Please ingest content first.", query_lang),
            "sources": [],
            "recommendations": [],
            "follow_ups": [],
            "session_id": session_id,
        }

    # Retrieve conversation history (passed to generator separately, not merged into context).
    conversation_history = ""
    if session_id:
        conversation_history = _conversation_manager.get_history(session_id)

    citations = _citations_from_hits(hits)
    generation = _qa_generator.generate_answer(
        sanitized_context,
        sanitized_query,
        sources=citations,
        learning_mode=learning_mode,
        conversation_history=conversation_history,
        user_mode=resolved_user_mode,
    )
    answer_text = _sanitize_output(str(generation.get("answer", "")))
    if query_lang != "en":
        answer_text = translate_from_english(answer_text, query_lang)
    formatted_sources = generation.get("sources") or []

    if session_id and answer_text:
        _conversation_manager.add_exchange(session_id, sanitized_query, answer_text)

    # Structured recommendations from generator + vector-based recommendations.
    recommender = Recommender(target=target)
    vector_recs = recommender.recommend(rewritten_query, top_k=min(3, top_k))
    llm_recs = generation.get("recommendations") or []
    # Merge: LLM recommendations first (topic-level), then vector recs (document-level).
    all_recs = llm_recs + vector_recs

    result = {
        "answer": answer_text,
        "sources": formatted_sources,
        "context": sanitized_context,
        "recommendations": all_recs,
        "follow_ups": generation.get("follow_ups") or [],
        "session_id": session_id,
        "learning_mode": learning_mode,
        "user_mode": resolved_user_mode,
        "ui_hints": get_ui_hints(resolved_user_mode),
    }
    # TASK 24: store in LRU cache (only for non-conversational queries).
    if cache_key:
        with _rag_cache_lock:
            _rag_cache[cache_key] = result
            if len(_rag_cache) > _RAG_CACHE_MAX:
                _rag_cache.popitem(last=False)
    return result


def get_document_chunks(document_id: str, target: str = "auto") -> tuple[str, List[str]]:
    """
    Fetch all chunks for a given document_id.

    The project stores the "document id" as `metadata["source"]` (filename/url/etc).
    Returns a tuple of (joined_text, chunks_list).
    """

    source = (document_id or "").strip()
    if not source:
        return "", []

    chunks: List[str] = []
    if _use_local(target):
        texts, _, metadatas = local_db.get_all_records()
        for text, meta in zip(texts, metadatas):
            if (meta.get("source") or meta.get("filename")) == source:
                chunks.append(text)

    if _use_qdrant(target):
        try:
            payloads = qdrant_store.payloads_by_source(source, limit=500)
            for payload in payloads:
                content = payload.get("content")
                if isinstance(content, str) and content.strip():
                    chunks.append(content)
        except Exception:
            pass

    joined = "\n\n".join(chunk.strip() for chunk in chunks if chunk and chunk.strip())
    return joined, chunks


def get_document_text(document_id: str, target: str = "auto") -> str:
    """Return the full document text for a given document_id (source)."""

    joined, _ = get_document_chunks(document_id, target=target)
    return joined


def summarize_text(text: str, max_length: int = 160) -> str:
    # TASK 4 FIX: delegate summarization to processing.summarization as the single source of truth.
    raw_summary = processing_summarize_text(text, max_length=max_length, min_length=60)
    return _sanitize_output(raw_summary)


def _resolve_chunking_strategy(modality: str, requested: str) -> str:
    """
    TASK 3: auto-select chunking when requested is 'auto' or default 'recursive'.

    PDF → recursive | Audio/Video transcript → fixed | URL → semantic | Code → fixed.
    Explicit fixed/semantic always wins; recursive/auto follow modality rules.
    """
    normalized = (requested or "recursive").strip().lower()
    if normalized not in {"auto", "recursive", ""}:
        return normalized
    if modality == "pdf":
        return "recursive"
    if modality in {"audio", "video"}:
        return "fixed"
    if modality == "url":
        return "semantic"
    if modality == "code":
        return "fixed"
    return "recursive"
