"""  
Sahayak AI — HuggingFace Model Registry (Singleton).

Loads and caches ML models on first use.  Every getter:
- Uses @lru_cache so the model is loaded exactly once per process
- Reads the model name from an env var (configurable)
- Returns None with a graceful fallback if loading fails

Models managed here:
  * Summarization  — facebook/bart-large-cnn
  * QnA            — deepset/roberta-base-squad2
  * Translation    — Helsinki-NLP/opus-mt-en-ROMANCE
  * Text generation — google/flan-t5-base (seq2seq via AutoModelForSeq2SeqLM)
  * Sentiment      — distilbert-base-uncased-finetuned-sst-2
  * Classification — facebook/bart-large-mnli
  * NER            — dbmdz/bert-large-cased-finetuned-conll03
  * Image class    — google/vit-base-patch16-224
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union
from dotenv import load_dotenv
load_dotenv()
if TYPE_CHECKING:
    # Only imported for type checkers — never at runtime. This keeps
    # torch/transformers OUT of the import chain entirely when
    # ENABLE_LOCAL_ML_MODELS=false (the default), which matters a lot on
    # low-RAM hosts: importing torch alone costs ~200-300MB before any
    # model is even loaded. On Render's 512MB free tier, that import cost
    # alone (combined with FastAPI + sentence-transformers + everything
    # else) was enough to hit the memory limit at startup, before a single
    # request came in.
    from transformers import Pipeline


# ---------------------------------------------------------------------------
# Lightweight seq2seq wrapper — replaces pipeline("text2text-generation")
# which fails on some transformers versions.
# ---------------------------------------------------------------------------

class Seq2SeqGenerator:
    """Wraps AutoTokenizer + AutoModelForSeq2SeqLM with a pipeline-like __call__."""

    def __init__(self, tokenizer, model) -> None:
        self._tokenizer = tokenizer
        self._model = model

    def __call__(
        self,
        text: str,
        max_length: int = 200,
        max_new_tokens: Optional[int] = None,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, str]]:
        """Generate text. Returns list of dicts with 'generated_text' key (pipeline-compatible)."""
        import torch  # lazy import — only reached when local models are enabled

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        # Move to same device as model
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        gen_kwargs: Dict[str, Any] = {"num_return_sequences": num_return_sequences}
        if max_new_tokens:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        else:
            gen_kwargs["max_length"] = max_length
        if do_sample:
            gen_kwargs["do_sample"] = True
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)
        results = []
        for out in outputs:
            decoded = self._tokenizer.decode(out, skip_special_tokens=True)
            results.append({"generated_text": decoded})
        return results


class HFModels:
    """Singleton class to load and manage HuggingFace models.

    IMPORTANT — memory footprint: these models range from ~250MB to ~1.6GB
    each when loaded. On low-RAM deployments (e.g. Render's free tier, 512MB
    total), loading even one of these will OOM the process. All getters are
    gated behind ENABLE_LOCAL_ML_MODELS (default: false). When disabled,
    getters return None immediately and callers should fall back to a hosted
    API (Groq/OpenAI) instead — see backend/routers/document_features.py and
    backend/services/knowledge_graph.py for the fallback pattern.
    Set ENABLE_LOCAL_ML_MODELS=true only in environments with enough RAM
    (a local dev machine, or a paid hosting tier).
    """

    _summarizer_pipeline: Optional["Pipeline"] = None
    _qna_pipeline: Optional["Pipeline"] = None
    _text_generation_pipeline: Optional["Pipeline"] = None
    _translation_pipeline: Optional["Pipeline"] = None

    @staticmethod
    def local_models_enabled() -> bool:
        return os.getenv("ENABLE_LOCAL_ML_MODELS", "false").strip().lower() == "true"

    # ------------------------------------------------------------------
    # Core models (Task 1)
    # ------------------------------------------------------------------

    @classmethod
    @lru_cache(maxsize=1)  # Ensure only one instance of the model is loaded
    def get_summarizer(cls) -> Optional["Pipeline"]:
        """Loads and returns the summarization pipeline (BART). None if disabled."""
        if not cls.local_models_enabled():
            return None
        from transformers import pipeline  # lazy import
        if cls._summarizer_pipeline is None:
            model_name = os.getenv("HF_MODEL_SUMMARIZER", "facebook/bart-large-cnn")
            try:
                cls._summarizer_pipeline = pipeline("summarization", model=model_name)
                print(f"Loaded summarizer model: {model_name}")
            except Exception as e:
                print(f"Error loading summarizer model {model_name}: {e}")
        return cls._summarizer_pipeline

    @classmethod
    @lru_cache(maxsize=1)
    def get_qna(cls) -> Optional["Pipeline"]:
        """Loads and returns the extractive QnA pipeline (RoBERTa-SQuAD)."""
        if not cls.local_models_enabled():
            return None
        from transformers import pipeline  # lazy import
        if cls._qna_pipeline is None:
            model_name = os.getenv("HF_MODEL_QNA", "deepset/roberta-base-squad2")
            try:
                cls._qna_pipeline = pipeline("question-answering", model=model_name)
                print(f"Loaded QnA model: {model_name}")
            except Exception as e:
                print(f"Error loading QnA model {model_name}: {e}")
        return cls._qna_pipeline

    @classmethod
    @lru_cache(maxsize=1)
    def get_text_generation(cls) -> Optional["Seq2SeqGenerator"]:
        """Loads Flan-T5 as a seq2seq model (NOT pipeline) for reliable text generation."""
        if not cls.local_models_enabled():
            return None
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # lazy import
        if cls._text_generation_pipeline is None:
            model_name = os.getenv("HF_MODEL_TEXT_GENERATION", "google/flan-t5-base")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                model.eval()
                cls._text_generation_pipeline = Seq2SeqGenerator(tokenizer, model)
                print(f"Loaded seq2seq text generation model: {model_name}")
            except Exception as e:
                print(f"Error loading text generation model {model_name}: {e}")
        return cls._text_generation_pipeline

    @classmethod
    @lru_cache(maxsize=1)
    def get_translation(cls) -> Optional["Pipeline"]:
        """Loads and returns the English→ROMANCE translation pipeline (Helsinki-NLP)."""
        if not cls.local_models_enabled():
            return None
        from transformers import pipeline  # lazy import
        if cls._translation_pipeline is None:
            model_name = os.getenv("HF_MODEL_TRANSLATION", "Helsinki-NLP/opus-mt-en-ROMANCE")
            try:
                cls._translation_pipeline = pipeline("translation", model=model_name)
                print(f"Loaded translation model: {model_name}")
            except Exception as e:
                print(f"Error loading translation model {model_name}: {e}")
        return cls._translation_pipeline

    # ------------------------------------------------------------------
    # Extended models (additional NLP capabilities)
    # ------------------------------------------------------------------

    _sentiment_pipeline: Optional["Pipeline"] = None
    _classification_pipeline: Optional["Pipeline"] = None
    _ner_pipeline: Optional["Pipeline"] = None
    _image_classification_pipeline: Optional["Pipeline"] = None

    @classmethod
    @lru_cache(maxsize=1)
    def get_sentiment(cls) -> Optional["Pipeline"]:
        """Loads and returns the sentiment analysis pipeline."""
        if not cls.local_models_enabled():
            return None
        from transformers import pipeline  # lazy import
        if cls._sentiment_pipeline is None:
            model_name = os.getenv("HF_MODEL_SENTIMENT", "distilbert-base-uncased-finetuned-sst-2-english")
            try:
                cls._sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
                print(f"Loaded sentiment model: {model_name}")
            except Exception as e:
                print(f"Error loading sentiment model {model_name}: {e}")
        return cls._sentiment_pipeline

    @classmethod
    @lru_cache(maxsize=1)
    def get_classification(cls) -> Optional["Pipeline"]:
        """Loads and returns the text classification pipeline."""
        if not cls.local_models_enabled():
            return None
        from transformers import pipeline  # lazy import
        if cls._classification_pipeline is None:
            model_name = os.getenv("HF_MODEL_CLASSIFICATION", "facebook/bart-large-mnli")
            try:
                cls._classification_pipeline = pipeline("zero-shot-classification", model=model_name)
                print(f"Loaded classification model: {model_name}")
            except Exception as e:
                print(f"Error loading classification model {model_name}: {e}")
        return cls._classification_pipeline

    @classmethod
    @lru_cache(maxsize=1)
    def get_ner(cls) -> Optional["Pipeline"]:
        """Loads and returns the named entity recognition pipeline."""
        if not cls.local_models_enabled():
            return None
        from transformers import pipeline  # lazy import
        if cls._ner_pipeline is None:
            model_name = os.getenv("HF_MODEL_NER", "dbmdz/bert-large-cased-finetuned-conll03-english")
            try:
                cls._ner_pipeline = pipeline("ner", model=model_name, aggregation_strategy="simple")
                print(f"Loaded NER model: {model_name}")
            except Exception as e:
                print(f"Error loading NER model {model_name}: {e}")
        return cls._ner_pipeline

    @classmethod
    @lru_cache(maxsize=1)
    def get_image_classification(cls) -> Optional["Pipeline"]:
        """Loads and returns the image classification pipeline."""
        if not cls.local_models_enabled():
            return None
        from transformers import pipeline  # lazy import
        if cls._image_classification_pipeline is None:
            model_name = os.getenv("HF_MODEL_IMAGE_CLASS", "google/vit-base-patch16-224")
            try:
                cls._image_classification_pipeline = pipeline("image-classification", model=model_name)
                print(f"Loaded image classification model: {model_name}")
            except Exception as e:
                print(f"Error loading image classification model {model_name}: {e}")
        return cls._image_classification_pipeline

# Example usage (for testing purposes, remove in production if not needed)
if __name__ == "__main__":
    # Set environment variables for testing different models or fallbacks
    # os.environ["HF_MODEL_SUMMARIZER"] = "sshleifer/distilbart-cnn-12-6"

    summarizer = HFModels.get_summarizer()
    if summarizer:
        text = "This is a long document that needs to be summarized. It contains many sentences and provides a lot of information."
        summary = summarizer(text, max_length=50, min_length=10, do_sample=False)
        print(f"\nSummary: {summary[0]['summary_text']}")
    else:
        print("Summarizer model not available.")

    qna = HFModels.get_qna()
    if qna:
        context = "The capital of France is Paris. It is a beautiful city."
        question = "What is the capital of France?"
        answer = qna(question=question, context=context)
        print(f"\nQnA Answer: {answer['answer']}")
    else:
        print("QnA model not available.")

    text_generator = HFModels.get_text_generation()
    if text_generator:
        prompt = "Once upon a time, in a land far, far away,"
        generated_text = text_generator(prompt, max_length=50, num_return_sequences=1)
        print(f"\nGenerated Text: {generated_text[0]['generated_text']}")
    else:
        print("Text generation model not available.")