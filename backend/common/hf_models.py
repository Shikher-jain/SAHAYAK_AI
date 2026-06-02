from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger("sahayak.hf_models")


@dataclass(frozen=True)
class HFModelConfig:
    """HuggingFace model ids used across Sahayak."""

    summarizer: str
    qna: str
    text_generator: str


def _resolve_config() -> HFModelConfig:
    """
    Resolve model ids from env vars with sensible defaults.

    Env vars:
    - HF_MODEL_SUMMARIZER
    - HF_MODEL_QNA
    - HF_MODEL_TEXT_GENERATOR
    """

    return HFModelConfig(
        summarizer=os.getenv("HF_MODEL_SUMMARIZER", "facebook/bart-large-cnn"),
        qna=os.getenv("HF_MODEL_QNA", "deepset/roberta-base-squad2"),
        text_generator=os.getenv("HF_MODEL_TEXT_GENERATOR", "google/flan-t5-base"),
    )


class _HFModelsSingleton:
    """
    Singleton HuggingFace pipelines.

    Pipelines are loaded lazily and cached to avoid repeated model loads.
    Each getter returns None when the model cannot be initialized, allowing
    callers to gracefully fall back.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._config = _resolve_config()
        self._summarizer: Any | None = None
        self._qna: Any | None = None
        self._text2text: Any | None = None
        self._attempted = {"summarizer": False, "qna": False, "text2text": False}

    def summarizer(self) -> Optional[Any]:
        """Return a HF summarization pipeline or None on failure."""

        if self._summarizer is not None:
            return self._summarizer
        with self._lock:
            if self._summarizer is not None:
                return self._summarizer
            if self._attempted["summarizer"]:
                return None
            self._attempted["summarizer"] = True
            try:
                from transformers import pipeline

                self._summarizer = pipeline("summarization", model=self._config.summarizer)
            except Exception as exc:
                logger.warning("HF summarizer init failed (%s): %s", self._config.summarizer, exc)
                self._summarizer = None
            return self._summarizer

    def qna(self) -> Optional[Any]:
        """Return a HF question-answering pipeline or None on failure."""

        if self._qna is not None:
            return self._qna
        with self._lock:
            if self._qna is not None:
                return self._qna
            if self._attempted["qna"]:
                return None
            self._attempted["qna"] = True
            try:
                from transformers import pipeline

                self._qna = pipeline("question-answering", model=self._config.qna)
            except Exception as exc:
                logger.warning("HF QnA init failed (%s): %s", self._config.qna, exc)
                self._qna = None
            return self._qna

    def text_generator(self) -> Optional[Any]:
        """Return a HF text2text-generation pipeline (flan-t5) or None on failure."""

        if self._text2text is not None:
            return self._text2text
        with self._lock:
            if self._text2text is not None:
                return self._text2text
            if self._attempted["text2text"]:
                return None
            self._attempted["text2text"] = True
            try:
                from transformers import pipeline

                self._text2text = pipeline("text2text-generation", model=self._config.text_generator)
            except Exception as exc:
                logger.warning("HF text generator init failed (%s): %s", self._config.text_generator, exc)
                self._text2text = None
            return self._text2text


_SINGLETON: _HFModelsSingleton | None = None
_SINGLETON_LOCK = Lock()


def get_hf_models() -> _HFModelsSingleton:
    """Get the shared HF models singleton."""

    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = _HFModelsSingleton()
        return _SINGLETON

