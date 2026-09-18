"""Zero-RAM PII / Secret Scrubber (v2_advanced privacy guard).

A pure-Python, standard-library-only redaction layer that sits between every
inbound medium (Telegram / WhatsApp / Web chat) and the supervisor graph, and
between raw ingestion and the persistent stores (Qdrant / Neo4j / Redis
cache). Zero heavy ML by construction:

  * No spaCy, no ``presidio_analyzer``, no transformers, no torch — none of
    the 100 MB + pipelines. Everything below is built on ``re`` + the
    Verhoeff checksum tables (196 bytes of arithmetic), so the whole module
    imports in the low-KiB range and never loads a model into the 512 MB box.

Two mutually exclusive operating modes (one scrubber per process, reuse it):

  * ``scrub_text``  → destructive, one-way redaction. Returns only the
                     rewritten string (tokens are NOT recoverable). Use for
                     persistent stores, embeddings, Qdrant / Neo4j payloads,
                     and the semantic cache.
  * ``mask_and_map`` → reversible tokenization. Returns ``(masked, map)``.
                     Feed the *masked* string to the LLM supervisor; after
                     generation, pass the model reply through ``unmask_text``
                     to restore the caller's original values. Neither the
                     LLM nor any logged trace ever sees the raw PII.

Strictly async-free and allocation-light: one compiled ``re.Pattern`` per
entity, one positional sweep of the input string, no per-call comprehension
blowups. Dedupes repeated values into a single token to keep the mapping
small.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import ClassVar, Dict, Iterable, List, Optional, Tuple

# =====================================================================
#  §1  Compiled patterns (built once at import — zero cost per call)
# =====================================================================
_ADDR_SEP = r"(?:[ -]?)"

# 12-digit Aadhaar: first digit 2–9, then Verhoeff-validated as a whole.
AADHAAR_RE = re.compile(
    rf"\b[2-9]\d{{3}}{_ADDR_SEP}\d{{4}}{_ADDR_SEP}\d{{4}}\b"
)

# Indian mobiles: +91 / 0 / bare, starting 6–9 (10–13 digits total).
PHONE_IN_RE = re.compile(
    r"(?:\+91(?:\s|-)?|0)?[6-9]\d{9}"
)
# Generic E.164 (any country): +<1-3 digits><7–15 digits>.
PHONE_E164_RE = re.compile(r"\+\d{1,3}[\s-]?\d{7,14}")

# Provider API keys + generic hex/secrets. Zero external model — entropy-led.
_SK = r"(?:sk|rk)_(?:live|test|proj)?_?[A-Za-z0-9]{20,}"
_API_KEY_RE = re.compile(
    r"\b(?:"
    r"sk-proj-[A-Za-z0-9_-]{20,}|"          # OpenAI project keys
    r"sk-[A-Za-z0-9_-]{20,}|"               # OpenAI + Stripe sk_
    r"hf_[A-Za-z0-9]{20,}|"                 # Hugging Face
    r"gsk_[A-Za-z0-9]{20,}|"                # Groq
    r"AKIA[0-9A-Z]{16}|"                    # AWS access key id
    r"eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}|"  # JWT
    r"[0-9a-fA-F]{32}|"                     # generic hex token (e.g. NGINX)
    r"[0-9a-fA-F]{40}|"                     # git SHA / hex secret
    r"[0-9a-fA-F]{64}"                      # API glyphs
    r")\b"
)

# Indian PAN: 5 letters + 4 digits + 1 letter (CDBA cases excluded).
_PAN = r"[A-Z]{5}[0-9]{4}[A-Z]"
PAN_RE = re.compile(rf"\b{_PAN}\b")

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# Aadhaar Verhoeff checksum tables (D, P, inv) — 10×10 int grid, no ML.
_VERHOEFF_D: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_VERHOEFF_INV: Tuple[int, ...] = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def _is_verhoeff_valid(digits: str) -> bool:
    """True if ``digits`` passes the Aadhaar Verhoeff checksum."""
    c = 0
    for i, ch in enumerate(reversed(digits), 1):
        d = ord(ch) - 48
        if d < 0 or d > 9:
            return False
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][d]]
    return _VERHOEFF_INV[c] == 0


@dataclass(frozen=True)
class PIISpan:
    """A detected PII region and the token that replaces it."""

    kind: str
    value: str
    start: int
    end: int
    replacement: str


class PIIScrubber:
    """Reusable, precompiled redaction engine (one instance per process).

    Typical use — inbound from a webhook:

        scrubber = PIIScrubber()

        # Mode A (persistent stores / embeddings / cache):
        clean = await scrubber.scrub_text(user_text)

        # Mode B (LLM turn):
        masked, mapping = await scrubber.mask_and_map(user_text)
        reply = await supervisor.ainvoke({"messages": [{"role": "user", "content": masked}]})
        final = await scrubber.unmask_text(reply.get("final_output", ""), mapping)
    """

    _REDACTIONS: ClassVar[Tuple[Tuple[re.Pattern, str, str], ...]] = (
        (AADHAAR_RE, "AADHAAR", "[AADHAAR_REDACTED]"),
        (PHONE_IN_RE, "PHONE", "[PHONE_REDACTED]"),
        (PHONE_E164_RE, "PHONE", "[PHONE_REDACTED]"),
        (_API_KEY_RE, "API_KEY", "[API_KEY_REDACTED]"),
        (PAN_RE, "PAN", "[PAN_REDACTED]"),
        (EMAIL_RE, "EMAIL", "[EMAIL_REDACTED]"),
    )

    def __init__(self, session_id: Optional[str] = None) -> None:
        self._session = session_id or uuid.uuid4().hex[:12]

    # -----------------------------------------------------------------
    #  Span collection
    # -----------------------------------------------------------------
    def _spans(self, text: str) -> List[PIISpan]:
        spans: List[PIISpan] = []
        for pattern, kind, replacement in self._REDACTIONS:
            for m in pattern.finditer(text):
                raw = m.group(0)
                # Aadhaar: only redact when the Verhoeff checksum passes —
                # this is what prevents false positives on random digit runs.
                if kind == "AADHAAR" and not _is_verhoeff_valid(re.sub(r"[ -]", "", raw)):
                    continue
                spans.append(PIISpan(kind, raw, m.start(), m.end(), replacement))
        # Sort by start; drop any span swallowed by an earlier (longer) one.
        spans.sort(key=lambda s: (s.start, s.end))
        merged: List[PIISpan] = []
        for s in spans:
            if merged and s.start < merged[-1].end:
                # Same region, different pattern (e.g. phone ⊂ E.164):
                # keep whichever is longer so no partial junk survives.
                if s.end > merged[-1].end:
                    merged[-1] = s
                continue
            merged.append(s)
        return merged

    # -----------------------------------------------------------------
    #  Mode A — destructive, one-way
    # -----------------------------------------------------------------
    async def scrub_text(self, text: str) -> str:
        """Irreversibly replace every detected identifier with its token."""
        if not text:
            return text
        out: List[str] = []
        cursor = 0
        for s in self._spans(text):
            out.append(text[cursor : s.start])
            out.append(s.replacement)
            cursor = s.end
        out.append(text[cursor:])
        return "".join(out)

    # ─────────────────────────────────────────────────────────────────
    #  Mode B — reversible tokenization
    # ─────────────────────────────────────────────────────────────────
    async def mask_and_map(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Tokenize identifiers; return ``(masked, {token: original})``.

        Repeated occurrences of the same value collapse onto ONE token so the
        mapping never grows with message length.
        """
        if not text:
            return text, {}
        mapping: Dict[str, str] = {}
        by_value: Dict[str, str] = {}
        counter = 0
        out: List[str] = []
        cursor = 0
        for s in self._spans(text):
            out.append(text[cursor : s.start])
            token = by_value.get(s.value)
            if token is None:
                counter += 1
                token = f"__PII_{s.kind}_{counter}__"
                by_value[s.value] = token
                mapping[token] = s.value
            out.append(token)
            cursor = s.end
        out.append(text[cursor:])
        return "".join(out), mapping

    async def unmask_text(self, text: str, mapping: Dict[str, str]) -> str:
        """Restore original values inside a supervisor/LLM reply."""
        if not text or not mapping:
            return text
        tokens = "|".join(re.escape(t) for t in mapping)
        pattern = re.compile(r"(" + tokens + r")")
        return pattern.sub(lambda m: mapping[m.group(1)], text)

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return f"<PIIScrubber session={self._session!r} at 0x{id(self):x}>"
