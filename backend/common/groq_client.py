"""Shared Groq chat-completion helper.

Groq's free tier is generous and fast, so it's the default backend for
tasks that would otherwise require loading a heavy local model (BART,
RoBERTa-QnA, Flan-T5, BERT-NER, ...). This keeps the deployed app's memory
footprint small enough to run on free-tier hosting (e.g. Render's 512MB
free web service) — see backend/common/hf_models.py for the local-model
gate this pairs with.

Existing call sites (rag/generator.py, rag/query_rewrite.py,
counselor_service.py) each have their own small Groq call already; this
module isn't a forced refactor of those (they work fine as-is) — it's for
new call sites (document_features.py, knowledge_graph.py) so we don't
duplicate the same 15 lines a fourth and fifth time.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


def groq_available() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def groq_complete(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Optional[str]:
    """Call Groq chat completions. Returns None (not an exception) on any
    failure — missing key, network error, bad response — so callers can
    fall back cleanly instead of crashing."""
    if not groq_available():
        return None
    try:
        from groq import Groq
    except ImportError:
        logger.warning("groq package not installed; cannot use Groq backend.")
        return None
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:
        logger.exception("Groq completion failed")
        return None

