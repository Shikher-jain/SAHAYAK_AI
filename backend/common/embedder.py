"""
Unified embedding module — single source of truth for Sahayak AI.

Priority: HuggingFace Inference API (hosted, no local RAM/disk cost) ->
local SentenceTransformer (only if ENABLE_LOCAL_ML_MODELS=true and you have
the RAM for it). This mirrors the same pattern used everywhere else in the
app (see backend/common/hf_models.py, hf_inference_api.py) — unlike those
optional NLP tasks, embeddings are NOT optional (RAG can't function without
them), which is exactly why this was the real cause of Render's memory
limit being hit even after every *optional* local model was gated off:
`sentence-transformers` pulls in `torch` unconditionally, and torch alone
costs ~200-300MB before any model is even loaded.

Configure model: EMBEDDING_MODEL env var (default: all-MiniLM-L6-v2).
All vectors are L2-normalized so cosine similarity equals the dot product.

NOTE ON HF INFERENCE API RESPONSE SHAPE: the feature-extraction endpoint's
response shape (already-pooled vector vs. per-token embeddings needing mean
pooling) can vary by model/API version — this is handled defensively below
(_to_pooled_vector), but if you see unexpected embedding dimensions after
deploying, verify the raw response shape with a manual curl call to the
endpoint and adjust _to_pooled_vector if needed.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
# Confirmed working via manual endpoint testing (2026-08) — HF migrated the
# Inference API to this router-based domain; the old api-inference.huggingface.co
# subdomain no longer resolves at all.
_HF_API_BASE = "https://router.huggingface.co/hf-inference/models"

# Singleton state for the LOCAL fallback model — only ever populated if
# ENABLE_LOCAL_ML_MODELS=true and the HF API path is unavailable/fails.
_MODEL = None
_MODEL_NAME: Optional[str] = None


def _local_models_enabled() -> bool:
    return os.getenv("ENABLE_LOCAL_ML_MODELS", "false").strip().lower() == "true"


def _hf_api_available() -> bool:
    return bool(os.getenv("HUGGINGFACEHUB_API_TOKEN"))


def _resolved_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL


def _hf_repo_id(model_name: str) -> str:
    """The HF Inference API needs a full repo id (e.g.
    'sentence-transformers/all-MiniLM-L6-v2'); the local SentenceTransformer
    loader accepts either the short or full name. Auto-prefix short names
    with the sentence-transformers org, since that's what EMBEDDING_MODEL
    is documented/defaulted to."""
    return model_name if "/" in model_name else f"sentence-transformers/{model_name}"


def _to_pooled_vector(response) -> Optional[List[float]]:
    """Normalize a feature-extraction API response for ONE input into a
    single flat embedding vector, handling both response shapes HF may
    return: already-pooled (flat list of floats) or per-token embeddings
    (list of lists, needing mean pooling)."""
    if not response:
        return None
    if isinstance(response[0], (int, float)):
        # Already a flat vector.
        return list(response)
    if isinstance(response[0], list):
        # Per-token embeddings — mean-pool across tokens.
        arr = np.asarray(response, dtype="float32")
        if arr.ndim == 2:
            return arr.mean(axis=0).tolist()
    return None


# def _hf_api_embed_one(text: str) -> Optional[List[float]]:
#     if not _hf_api_available():
#         return None
#     try:
#         import requests
#     except ImportError:
#         return None
#     model_name = _hf_repo_id(_resolved_model_name())
#     headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN', '')}"}
#     try:
#         resp = requests.post(
#             f"{_HF_API_BASE}/{model_name}",
#             headers=headers,
#             json={"inputs": text},
#             timeout=30,
#         )
#         if resp.status_code == 503:
#             # Model cold-starting on HF's side — retry once with wait flag.
#             resp = requests.post(
#                 f"{_HF_API_BASE}/{model_name}",
#                 headers=headers,
#                 json={"inputs": text, "options": {"wait_for_model": True}},
#                 timeout=60,
#             )
#         resp.raise_for_status()
#         return _to_pooled_vector(resp.json())
#     except Exception:
#         return None


def _hf_api_embed_one(text: str) -> Optional[List[float]]:
    if not _hf_api_available():
        # print("DEBUG embedder: HUGGINGFACEHUB_API_TOKEN not set")
        return None
    try:
        import requests
    except ImportError:
        return None
    model_name = _hf_repo_id(_resolved_model_name())
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN', '')}"}
    url = f"{_HF_API_BASE}/{model_name}/pipeline/feature-extraction"
    try:
        resp = requests.post(url, headers=headers, json={"inputs": text}, timeout=30)
        if resp.status_code == 503:
            resp = requests.post(
                url, headers=headers,
                json={"inputs": text, "options": {"wait_for_model": True}},
                timeout=60,
            )
        # print(f"DEBUG embedder: url={url} status={resp.status_code} body={resp.text[:500]}")
        resp.raise_for_status()
        parsed = _to_pooled_vector(resp.json())
        # print(f"DEBUG embedder: parsed vector is None? {parsed is None}")
        return parsed
    except Exception as exc:
        # print(f"DEBUG embedder: exception = {type(exc).__name__}: {exc}")
        return None
        
def _get_local_model():
    """Load the local SentenceTransformer (fallback path only)."""
    global _MODEL, _MODEL_NAME
    from sentence_transformers import SentenceTransformer  # lazy import

    model_name = _resolved_model_name()
    if _MODEL is None or _MODEL_NAME != model_name:
        _MODEL = SentenceTransformer(
            model_name,
            device="cpu",
            model_kwargs={"low_cpu_mem_usage": False},
        )
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
    """Embed a single string; output is float32 and L2-normalized.

    Priority: HF Inference API -> local model (only if ENABLE_LOCAL_ML_MODELS=true).
    Raises RuntimeError if neither is available — unlike the optional NLP
    features elsewhere in the app, there's no safe silent fallback here:
    without an embedding, ingestion/search can't function at all, and
    returning a zero/garbage vector would silently corrupt the vector store.
    """
    api_vector = _hf_api_embed_one(text)
    if api_vector is not None:
        return _normalize_vectors(np.asarray(api_vector, dtype="float32"))

    if _local_models_enabled():
        model = _get_local_model()
        vector = np.asarray(model.encode(text, show_progress_bar=False), dtype="float32")
        return _normalize_vectors(vector)

    raise RuntimeError(
        "No embedding backend available: HF Inference API failed/unconfigured "
        "(check HUGGINGFACEHUB_API_TOKEN) and local models are disabled "
        "(ENABLE_LOCAL_ML_MODELS=false). Embeddings are required for ingestion "
        "and search to function."
    )


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Batch-embed many strings.

    NOTE: calls the HF Inference API once per text rather than one batched
    call — the batched response shape (pooled vs. per-token, and how it
    nests for multiple inputs) isn't reliably predictable across models/API
    versions, and getting that wrong would silently corrupt embeddings.
    This trades some latency for correctness; if this becomes a real
    bottleneck for large documents, verify the live batched response shape
    first, then revisit.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    if _hf_api_available():
        vectors = [_hf_api_embed_one(t) for t in texts]
        if all(v is not None for v in vectors):
            return _normalize_vectors(np.asarray(vectors, dtype="float32"))
        # Partial/total API failure — fall through to local if enabled.

    if _local_models_enabled():
        model = _get_local_model()
        vectors = np.asarray(model.encode(texts, show_progress_bar=False), dtype="float32")
        return _normalize_vectors(vectors)

    raise RuntimeError(
        "No embedding backend available: HF Inference API failed/unconfigured "
        "(check HUGGINGFACEHUB_API_TOKEN) and local models are disabled "
        "(ENABLE_LOCAL_ML_MODELS=false). Embeddings are required for ingestion "
        "and search to function."
    )