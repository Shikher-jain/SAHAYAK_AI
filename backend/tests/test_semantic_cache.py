"""Offline tests for the Upstash semantic cache layer (v2_advanced).

Guarantees:
  * ZERO network — every `AsyncIndex` call is an `unittest.mock.AsyncMock`.
  * ZERO local models — nothing imports torch / sentence-transformers; the
    Upstash index handles embedding server-side ("data" string -> configured
    model). No `upstash_vector` SDK is even required on disk to *run* these:
    `AsyncIndex.__new__` is replaced before the real module is imported so the
    class can construct + the tests can mock, regardless of env credentials.
  * Async-first — every test is `async def` with `pytest.mark.asyncio`.

Run:  pytest backend/tests/test_semantic_cache.py -v
"""
import copy
import json
import os
import sys
import time
import types
from typing import Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio

# ─────────────────────────────────────────────────────────────────────────
# Hard pre-import: swap out the *class* before `semantic_cache.py` executes
# `from upstash_vector import AsyncIndex`, so a missing/absent SDK neither
# blocks collection NOR lets a real index be hand-rolled in tests.
# ─────────────────────────────────────────────────────────────────────────
FAKE_UPSTASH_MODULE = types.ModuleType("upstash_vector")

class _FakeAsyncIndex:
    """Stand-in whose instances are always AsyncMocks keyed to the call sites
    the cache really uses (`query(data=…)` for reads, `upsert(…)` for writes)."""

    def __new__(cls, *args, **kwargs):  # noqa: ANN002, ANN003 — __new__ signature
        instance = MagicMock()
        instance.query = AsyncMock(return_value=[])
        instance.upsert = AsyncMock()
        return instance

FAKE_UPSTASH_MODULE.AsyncIndex = _FakeAsyncIndex
sys.modules[__name__] = sys.modules[__name__]  # ensure import machinery stays well-ordered
sys.modules["upstash_vector"] = FAKE_UPSTASH_MODULE

# Now the real module imports cleanly (constructor sees no env creds → disabled
# → get/set short-circuit to None, which is exactly the miss/bypass contract).
from backend.cache.semantic_cache import SemanticCache  # noqa: E402
from backend.cache import semantic_cache as sc_mod          # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def cache_service() -> SemanticCache:
    svc = SemanticCache()
    return svc


@pytest.fixture
def given_index(cache_service):
    """A fake index bound onto a fresh service — replaces the module scoping
    the old suite relied on, so tests are fully independent & parallel-safe."""
    cs = SemanticCache()
    cs.index = _FakeAsyncIndex()
    return cs, cs.index


# --- READ PATH: semantic hit / miss / empty ---------------------------------
async def test_cache_hit_returns_payload_within_threshold(given_index):
    cs, idx = given_index
    ok = MagicMock()
    ok.score = 0.98
    ok.metadata = {
        "response": "Refunds are processed within 5–7 business days.",
        "agent_type": "rag_agent",
        "citations": json.dumps([{"page_number": 12, "citation_index": 1}]),
        "timestamp": time.time(),
    }
    idx.query.return_value = [ok]

    result = await cs.get_cached_response(
        "how long does a refund take?", user_id="user_123"
    )

    assert result is not None
    assert result["cached"] is True
    assert result["response"] == "Refunds are processed within 5–7 business days."
    assert result["routed_agent"] == "rag_agent"
    assert isinstance(result["citations"], list)
    assert result["citations"][0]["page_number"] == 12
    idx.query.assert_awaited_once()


async def test_cache_miss_below_similarity_threshold(given_index):
    cs, idx = given_index
    low = MagicMock()
    low.score = 0.85          # just under 0.92 → must NOT be returned
    low.metadata = {"response": "…", "timestamp": time.time()}
    idx.query.return_value = [low]

    result = await cs.get_cached_response("some unrelated topic", user_id="user_9")

    assert result is None
    idx.query.assert_awaited_once()


async def test_cache_miss_on_empty_results(given_index):
    cs, idx = given_index
    idx.query.return_value = []

    result = await cs.get_cached_response("brand new question", user_id="user_9")

    assert result is None
    idx.query.assert_awaited_once()


async def test_cache_hit_stale_ttl_returns_none(given_index):
    cs, idx = given_index
    stale = MagicMock()
    stale.score = 0.99
    stale.metadata = {
        "response": "old",
        "agent_type": "rag_agent",
        "citations": "[]",
        "timestamp": time.time() - 10 * 24 * 3600,   # older than default 2-day TTL
    }
    idx.query.return_value = [stale]

    result = await cs.get_cached_response("age + very old q", user_id="user_9")

    assert result is None


# --- WRITE PATH / SAFETY GATES ------------------------------------------------
async def test_counseling_agent_response_is_never_cached(given_index):
    """Rule: counseling-agent routed output MUST NOT enter the vector store."""
    cs, idx = given_index
    payload = {
        "response": "You are not alone… 14416.",
        "routed_agent": "counseling_agent",
    }

    await cs.set_cached_response(
        "I feel so depressed and need immediate help", response_data=payload
    )

    idx.upsert.assert_not_awaited()


async def test_distress_query_bypasses_cache_read(cache_service):
    """Crisis keywords short-circuit the read — no index.query is ever made."""
    idx = _FakeAsyncIndex()
    cache_service.index = idx

    result = await cache_service.get_cached_response(
        "i feel like killing myself", "user_123"
    )

    assert result is None
    idx.query.assert_not_awaited()


async def test_personal_ephemeral_query_bypasses_cache_read(cache_service):
    """Session-specific phrasing ('my uploaded…') must not hit the cache."""
    idx = _FakeAsyncIndex()
    cache_service.index = idx

    result = await cache_service.get_cached_response(
        "summarize my uploaded document", "user_123"
    )

    assert result is None
    idx.query.assert_not_awaited()


async def test_cache_set_builds_upstash_payload(cache_service):
    """Verify upsert carries an embedding-ready data string + JSON metadata."""
    idx = _FakeAsyncIndex()
    cache_service.index = idx

    payload = {
        "response": "Upstash Vector REST caches server-side embeddings.",
        "routed_agent": "rag_agent",
        "citations": [{"page_number": 3, "citation_index": 1}],
    }
    await cache_service.set_cached_response("what is semantic caching", payload)

    idx.upsert.assert_awaited_once()
    call = idx.upsert.call_args
    vectors = call.kwargs.get("vectors") or call.args[0]
    assert len(vectors) == 1
    vector = vectors[0]
    assert vector["data"] == "what is semantic caching"
    assert vector["metadata"]["response"] == payload["response"]
    assert vector["metadata"]["agent_type"] == "rag_agent"
    assert json.loads(vector["metadata"]["citations"])[0]["page_number"] == 3


async def test_cache_set_does_not_store_routing_only(cache_service):
    """unknown agent responses are intentionally not stored."""
    idx = _FakeAsyncIndex()
    cache_service.index = idx

    await cache_service.set_cached_response(
        "code of conduct question",
        {"response": "(no answer)", "routed_agent": "unknown"},
    )

    idx.upsert.assert_not_awaited()
