"""Language detection + translation.

Priority: HF Inference API (hosted, no local RAM/disk cost) -> local
Helsinki-NLP model (only if ENABLE_LOCAL_ML_MODELS=true) -> original text
unchanged (silent, safe fallback — never crashes, just skips translation).

Same router.huggingface.co pattern already confirmed working for embeddings,
summarization, QA, NER, and ASR this session. Translation's exact task-name
("translation") follows HF's standard taxonomy but — unlike those five —
hasn't been individually empirically verified yet; if a call 400s, use the
same diagnostic-script approach used earlier to confirm the exact URL/params.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from langdetect import detect, DetectorFactory
from dotenv import load_dotenv
load_dotenv()
# Ensure reproducibility in langdetect
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = ["hi", "en", "es", "fr", "de"]

_HF_API_BASE = "https://router.huggingface.co/hf-inference/models"


def _local_models_enabled() -> bool:
    return os.getenv("ENABLE_LOCAL_ML_MODELS", "false").strip().lower() == "true"


def _hf_api_available() -> bool:
    return bool(os.getenv("HUGGINGFACEHUB_API_TOKEN"))


def _hf_api_translate(text: str, model_name: str) -> Optional[str]:
    """Tier 1: HF Inference API, language-pair-specific model (same models
    used locally — e.g. Helsinki-NLP/opus-mt-hi-en — just called remotely
    instead of loaded into RAM)."""
    if not _hf_api_available():
        return None
    try:
        import requests
    except ImportError:
        return None
    url = f"{_HF_API_BASE}/{model_name}/pipeline/translation"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN', '')}"}
    try:
        resp = requests.post(url, headers=headers, json={"inputs": text[:4000]}, timeout=30)
        if resp.status_code == 503:
            resp = requests.post(
                url, headers=headers,
                json={"inputs": text[:4000], "options": {"wait_for_model": True}},
                timeout=60,
            )
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list) and result and "translation_text" in result[0]:
            return result[0]["translation_text"]
        return None
    except Exception as e:
        print(f"HF API translation failed for {model_name}: {e}")
        return None


@lru_cache(maxsize=None)  # Cache translation pipelines indefinitely
def _get_local_translation_pipeline(model_name: str):
    """Tier 2: local model, only if ENABLE_LOCAL_ML_MODELS=true."""
    if not _local_models_enabled():
        return None
    try:
        from transformers import pipeline
        translator = pipeline("translation", model=model_name)
        print(f"Loaded translation model: {model_name}")
        return translator
    except Exception as e:
        print(f"Error loading translation model {model_name}: {e}")
        return None


def detect_language(text: str) -> str:
    """Detects the language of the given text and returns its ISO 639-1 code."""
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED_LANGUAGES else "en"
    except Exception as e:
        print(f"Language detection failed: {e}. Defaulting to English.")
        return "en"


def _translate(text: str, model_name: str) -> str:
    """Shared HF-API -> local -> unchanged fallback chain."""
    api_result = _hf_api_translate(text, model_name)
    if api_result:
        return api_result

    translator = _get_local_translation_pipeline(model_name)
    if translator:
        try:
            return translator(text)[0]["translation_text"]
        except Exception as e:
            print(f"Local translation failed for {model_name}: {e}. Returning original text.")

    return text  # Both tiers unavailable/failed — safe, silent fallback.


def translate_to_english(text: str, source_lang: str) -> str:
    """Translates text from source_lang to English."""
    if source_lang == "en":
        return text
    return _translate(text, f"Helsinki-NLP/opus-mt-{source_lang}-en")


def translate_from_english(text: str, target_lang: str) -> str:
    """Translates text from English to target_lang."""
    if target_lang == "en":
        return text
    return _translate(text, f"Helsinki-NLP/opus-mt-en-{target_lang}")