from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional

logger = logging.getLogger("sahayak.language")


SUPPORTED_LANGS = {"en", "hi", "es", "fr", "de"}


def detect_language(text: str) -> str:
    """
    Detect language code for a given text.

    Returns a best-effort ISO code; falls back to "en" on errors/empty input.
    """

    if not text or not text.strip():
        return "en"
    try:
        from langdetect import detect

        lang = detect(text)
        return lang if lang in SUPPORTED_LANGS else "en"
    except Exception:
        return "en"


@dataclass(frozen=True)
class _TranslatorConfig:
    """Supported translation model naming for opus-mt."""

    model_prefix: str = "Helsinki-NLP/opus-mt"


class _TranslationPipelines:
    """
    Lazy-loaded translation pipelines by model name.

    This caches pipelines to avoid re-downloading/re-initializing the same model.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._pipelines: Dict[str, object] = {}
        self._failed: set[str] = set()
        self._cfg = _TranslatorConfig()

    def _model_name(self, source_lang: str, target_lang: str) -> str:
        return f"{self._cfg.model_prefix}-{source_lang}-{target_lang}"

    def get(self, source_lang: str, target_lang: str) -> Optional[object]:
        """Return a translation pipeline for the pair, or None if unavailable."""

        model_name = self._model_name(source_lang, target_lang)
        if model_name in self._pipelines:
            return self._pipelines[model_name]
        if model_name in self._failed:
            return None
        with self._lock:
            if model_name in self._pipelines:
                return self._pipelines[model_name]
            if model_name in self._failed:
                return None
            try:
                from transformers import pipeline

                pipe = pipeline("translation", model=model_name)
                self._pipelines[model_name] = pipe
                return pipe
            except Exception as exc:
                logger.warning("Translation model init failed (%s): %s", model_name, exc)
                self._failed.add(model_name)
                return None


_TRANSLATORS: _TranslationPipelines | None = None
_TRANSLATORS_LOCK = Lock()


def _get_translators() -> _TranslationPipelines:
    """Return the shared translation cache singleton."""

    global _TRANSLATORS
    if _TRANSLATORS is not None:
        return _TRANSLATORS
    with _TRANSLATORS_LOCK:
        if _TRANSLATORS is None:
            _TRANSLATORS = _TranslationPipelines()
        return _TRANSLATORS


def translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate input text to English.

    If translation pipeline cannot be loaded, returns original text unchanged.
    """

    src = (source_lang or "en").lower()
    if src == "en" or not text.strip():
        return text
    if src not in SUPPORTED_LANGS:
        return text

    translators = _get_translators()
    pipe = translators.get(src, "en")
    if pipe is None:
        return text
    try:
        result = pipe(text)
        if isinstance(result, list) and result:
            return str(result[0].get("translation_text", "")).strip() or text
        return text
    except Exception:
        return text


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate English text into the target language.

    If translation pipeline cannot be loaded, returns original text unchanged.
    """

    tgt = (target_lang or "en").lower()
    if tgt == "en" or not text.strip():
        return text
    if tgt not in SUPPORTED_LANGS:
        return text

    translators = _get_translators()
    pipe = translators.get("en", tgt)
    if pipe is None:
        return text
    try:
        result = pipe(text)
        if isinstance(result, list) and result:
            return str(result[0].get("translation_text", "")).strip() or text
        return text
    except Exception:
        return text

