"""
Unified embedding module — single source of truth for Sahayak AI.

Singleton: the SentenceTransformer model is loaded once via get_model() and
reused across ingestion, RAG, local stack, and finetune paths.

Configure model: EMBEDDING_MODEL env var (default: all-MiniLM-L6-v2).
All vectors are L2-normalized so cosine similarity equals the dot product.
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

# Singleton state — one model instance per process.
_MODEL: SentenceTransformer | None = None
_MODEL_NAME: str | None = None

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


def get_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance (loads on first use)."""
    global _MODEL, _MODEL_NAME
    model_name = os.getenv("EMBEDDING_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    if _MODEL is None or _MODEL_NAME != model_name:
        _MODEL = SentenceTransformer(model_name)
        _MODEL_NAME = model_name
    return _MODEL


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors in-place semantics; zero vectors are left unchanged."""
    if vectors.size == 0:
        return vectors
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors if norm == 0 else vectors / norm
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_text(text: str) -> np.ndarray:
    """Embed a single string; output is float32 and L2-normalized."""
    model = get_model()
    vector = np.asarray(model.encode(text, show_progress_bar=False), dtype="float32")
    return _normalize_vectors(vector)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Batch-embed many strings in one model.encode() call.

    Much faster than looping embed_text() for large documents (see vector_service).
    Returns shape (n, dim) with L2-normalized rows.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    model = get_model()
    vectors = np.asarray(model.encode(texts, show_progress_bar=False), dtype="float32")
    return _normalize_vectors(vectors)
