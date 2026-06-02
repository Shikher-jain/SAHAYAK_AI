from langdetect import detect, DetectorFactory
from transformers import pipeline, Pipeline
from functools import lru_cache
import os
from typing import Optional

# Ensure reproducibility in langdetect
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = ["hi", "en", "es", "fr", "de"]

@lru_cache(maxsize=None) # Cache translation pipelines indefinitely
def _get_translation_pipeline(model_name: str) -> Optional[Pipeline]:
    """Helper to load a translation pipeline with caching and error handling."""
    try:
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
        return lang if lang in SUPPORTED_LANGUAGES else "en" # Default to English if unsupported
    except Exception as e:
        print(f"Language detection failed: {e}. Defaulting to English.")
        return "en"

def translate_to_english(text: str, source_lang: str) -> str:
    """Translates text from source_lang to English using Helsinki-NLP/opus-mt models."""
    if source_lang == "en":
        return text
    
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-en"
    translator = _get_translation_pipeline(model_name)
    if translator:
        try:
            translated_text = translator(text)[0]["translation_text"]
            return translated_text
        except Exception as e:
            print(f"Translation to English failed for {source_lang}: {e}. Returning original text.")
            return text
    return text # Fallback to original text if model fails to load

def translate_from_english(text: str, target_lang: str) -> str:
    """Translates text from English to target_lang using Helsinki-NLP/opus-mt models."""
    if target_lang == "en":
        return text
    
    model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
    translator = _get_translation_pipeline(model_name)
    if translator:
        try:
            translated_text = translator(text)[0]["translation_text"]
            return translated_text
        except Exception as e:
            print(f"Translation from English failed for {target_lang}: {e}. Returning original text.")
            return text
    return text # Fallback to original text if model fails to load

if __name__ == "__main__":
    print(f"Detected language of 'Hello world': {detect_language('Hello world')}")
    print(f"Detected language of 'Hola mundo': {detect_language('Hola mundo')}")
    print(f"Detected language of 'नमस्ते दुनिया': {detect_language('नमस्ते दुनिया')}")

    english_text = "Hello, how are you?"
    spanish_text = "Hola, ¿cómo estás?"
    hindi_text = "नमस्ते, आप कैसे हैं?"

    print(f"\nTranslating '{spanish_text}' to English: {translate_to_english(spanish_text, 'es')}")
    print(f"Translating '{hindi_text}' to English: {translate_to_english(hindi_text, 'hi')}")
    print(f"Translating '{english_text}' to Spanish: {translate_from_english(english_text, 'es')}")
    print(f"Translating '{english_text}' to Hindi: {translate_from_english(english_text, 'hi')}")