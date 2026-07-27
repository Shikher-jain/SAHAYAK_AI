from langdetect import detect, DetectorFactory
from functools import lru_cache
import os
from typing import Optional

# Ensure reproducibility in langdetect
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = ["hi", "en", "es", "fr", "de"]


def _local_models_enabled() -> bool:
    return os.getenv("ENABLE_LOCAL_ML_MODELS", "false").strip().lower() == "true"


@lru_cache(maxsize=None)  # Cache translation pipelines indefinitely
def _get_translation_pipeline(model_name: str):
    """Helper to load a translation pipeline with caching and error handling.
    Only attempts to load if ENABLE_LOCAL_ML_MODELS=true — otherwise returns
    None immediately, so translation silently falls back to returning the
    original text (see translate_to_english/translate_from_english below)
    instead of downloading a 300MB+ model mid-request."""
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


def translate_to_english(text: str, source_lang: str) -> str:
    """Translates text from source_lang to English using Helsinki-NLP/opus-mt models.
    Returns the original text unchanged if local models are disabled — this is
    a silent, safe fallback (no translation, not a crash)."""
    if source_lang == "en":
        return text
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-en"
    translator = _get_translation_pipeline(model_name)
    if translator:
        try:
            return translator(text)[0]["translation_text"]
        except Exception as e:
            print(f"Translation to English failed for {source_lang}: {e}. Returning original text.")
    return text


def translate_from_english(text: str, target_lang: str) -> str:
    """Translates text from English to target_lang. Same fallback behavior as above."""
    if target_lang == "en":
        return text
    model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
    translator = _get_translation_pipeline(model_name)
    if translator:
        try:
            return translator(text)[0]["translation_text"]
        except Exception as e:
            print(f"Translation from English failed for {target_lang}: {e}. Returning original text.")
    return text