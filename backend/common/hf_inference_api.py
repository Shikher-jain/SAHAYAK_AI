"""HuggingFace Inference API — hosted model calls, no local RAM/disk cost.

Replaces local pipeline loading (backend/common/hf_models.py's old default
behavior) with HTTP calls to HF's hosted Inference API. This is the second
tier in the fallback chain: Groq (fast, general LLM) -> here (specialized
task-specific models, no local footprint) -> local model (only if you
explicitly set ENABLE_LOCAL_ML_MODELS=true and have the RAM for it).

Requires HUGGINGFACEHUB_API_TOKEN (free — https://huggingface.co/settings/tokens).

NOTE: free-tier serverless Inference API availability/limits/response shapes
have changed over time on HF's side. If a call here starts failing, check
https://huggingface.co/docs/api-inference for current behavior before
assuming this code is wrong — verify against a live curl/Postman call to
the specific model endpoint first.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)

# Confirmed working via manual endpoint testing (2026-08) — HF migrated the
# Inference API to this router-based domain; the old api-inference.huggingface.co
# subdomain no longer resolves at all.
_API_BASE = "https://router.huggingface.co/hf-inference/models"


def hf_api_available() -> bool:
    return bool(os.getenv("HUGGINGFACEHUB_API_TOKEN"))


def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN', '')}"}


def _post(model: str, task: str, payload: Dict[str, Any], timeout: int = 30) -> Optional[Any]:
    """POST to a HF Inference API model endpoint for a specific task.
    Returns None (not an exception) on any failure — missing token, network
    error, model loading/cold-start error, bad response — so callers can
    fall back cleanly.

    NOTE: task names (summarization, question-answering, token-classification,
    text2text-generation) follow HF's standard pipeline taxonomy but — unlike
    feature-extraction, which was empirically verified — haven't each been
    individually confirmed against the live API. If one of these 404s,
    re-run the same diagnostic-script approach used to find the working
    embeddings URL to confirm the exact task string for that model.
    """
    if not hf_api_available():
        return None
    try:
        import requests
    except ImportError:
        logger.warning("requests package not installed; cannot call HF Inference API.")
        return None
    url = f"{_API_BASE}/{model}/pipeline/{task}"
    try:
        resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout)
        if resp.status_code == 503:
            logger.info("HF model %s is cold-starting, retrying with wait_for_model...", model)
            payload = {**payload, "options": {"wait_for_model": True}}
            resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout + 30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.exception("HF Inference API call failed for model %s (task=%s)", model, task)
        return None

def hf_api_summarize(text: str) -> Optional[str]:
    model = os.getenv("HF_MODEL_SUMMARIZER", "facebook/bart-large-cnn")
    result = _post(model, "summarization", {"inputs": text[:4000]})
    try:
        return result[0]["summary_text"].strip()
    except (TypeError, KeyError, IndexError):
        return None


def hf_api_qna(question: str, context: str) -> Optional[Dict[str, Any]]:
    model = os.getenv("HF_MODEL_QNA", "deepset/roberta-base-squad2")
    result = _post(model, "question-answering", {"inputs": {"question": question, "context": context[:4000]}})
    if isinstance(result, dict) and "answer" in result:
        return {"answer": result["answer"], "score": result.get("score", 0.5)}
    return None


def hf_api_generate(prompt: str, max_new_tokens: int = 300) -> Optional[str]:
    model = os.getenv("HF_MODEL_TEXT_GENERATION", "google/flan-t5-base")
    result = _post(model, "text2text-generation", {"inputs": prompt[:4000], "parameters": {"max_new_tokens": max_new_tokens}})
    try:
        return result[0]["generated_text"].strip()
    except (TypeError, KeyError, IndexError):
        return None


def hf_api_ner(text: str) -> Optional[List[str]]:
    model = os.getenv("HF_MODEL_NER", "dbmdz/bert-large-cased-finetuned-conll03-english")
    result = _post(model, "token-classification", {"inputs": text[:2000], "parameters": {"aggregation_strategy": "simple"}})
    try:
        entities = list({item["word"].strip() for item in result if item.get("word")})
        return entities[:20] or None
    except (TypeError, KeyError):
        return None