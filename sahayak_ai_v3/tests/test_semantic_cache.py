"""Offline tests for `SemanticCache` (the v2_advanced semantic-cache layer).

This suite exercises the *real* `SemanticCache` class from
`backend/cache/semantic_cache.py` with every Upstash call replaced by
`unittest.mock.AsyncMock` — so it runs:

  * with ZERO network I/O (Upstash is never contacted),
  * with ZERO local compute (no torch / sentence-transformers / embedding
    models, no `upstash_vector` SDK even present on disk),
  * fully asynchronously (pytest-asyncio).

Contract exercised (matches semantic_cache.py):

    SemanticCache()
        enabled=False  when  UPSTASH_VECTOR_REST_URL/_TOKEN  missing
        index         →  upstash_vector.AsyncIndex
    async get_cached_response(query, user_id) -> Optional[dict]
        bypasses   crisis / session-personal queries           → None
        no results / score < 0.92 / TTL expired                → None
        hit                                               → parsed payload
    async set_cached_response(query, response_data)
        bypasses   crisis / session-personal queries           (no write)
        skips      routed_agent in {counseling_agent, unknown} (no write)
        writes     index.upsert(vectors=[{id, data, metadata}])

To keep the SDK off the machine entirely, `AsyncIndex` is stubbed *before* the
module is imported via a tiny fake `upstash_vector` module in `sys.modules`.
"""
import json
import os
import sys
import time
import types
from typing import AsyncIterator, Dict, Optional, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio


# ─────────────────────────────────────────────────────────────────────────────
# Importable SDK shim — prepopulated *before* semantic_cache is imported.
# Tests then swap in `AsyncMock` instances for the index itself, so the real
# class body is exercised without a live Upstash / SDK / embedding model.
# ─────────────────────────────────────────────────────────────────────────────
_fake_upstash = types.ModuleType("upstash_vector")


class _FakeAsyncIndex:
    """Stand-in for `upstash_vector.index.AsyncIndex` — identical constructor
    shape, never touches the network."""

    def __init__(self, url: str = "", token: str = ""):
        self.url = url
        self.token = token


_fake_upstash.AsyncIndex = _FakeAsyncIndex
sys.modules.setdefault("upstash_vector", _fake_upstash)

# Only now is the real module safe to import.
from backend.cache.semantic_cache import (  # noqa: E402
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
    SEMANTIC_CACHE_TTL_SECONDS,
    SemanticCache,
    semantic_cache,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def mock_upstash_index() -> AsyncMock:
    """An AsyncMock dressed as the Upstash index: `.query` returns [] by
    default, `.upsert` is awaited. The *real* semantic-cache module binds to
    this at construction time via its `index` attribute."""
    index = AsyncMock()
    index.query = AsyncMock(return_value=[])
    index.upsert = AsyncMock()
    yield index


@pytest.fixture
def cache_service(mock_upstash_index: AsyncMock) -> SemanticCache:
    svc = SemanticCache()
    svc.index = mock_upstash_index
    svc.enabled = True
    return svc


def _query_hit(score: float, metadata: Dict[str, str], **extra: Any) -> MagicMock:
    """A fake solver result carrying `.score` and `.metadata` (and whatever
    else the index layer appends, e.g. `.id`)."""
    res = AsyncMock()
    res.score = score
    res.metadata = dict(metadata)
    for k, v in extra.items():
        setattr(res, k, v)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# READ PATH — HIT / MISS / THRESHOLD GATE
# ─────────────────────────────────────────────────────────────────────────────
async def test_cache_hit_exact_match(cache_service, mock_upstash_index):
    """A semantically-identical answer → parsed payload with a cached flag."""
    mock_hit = _query_hit(
        0.98,
        {
            "response": "Refunds are issued within 5 business days.",
            "agent_type": "rag_agent",
            "citations": json.dumps([{"page_number": 12, "citation_index": 1}]),
            "timestamp": int(time.time()),
        },
    )
    mock_upstash_index.query.return_value = [mock_hit]

    result = await cache_service.get_cached_response(
        "What is the refund policy?", "user_123"
    )

    assert result is not None
    assert result["response"] == "Refunds are issued within 5 business days."
    assert result["routed_agent"] == "rag_agent"
    assert len(result["citations"]) == 1
    assert result["citations"][0]["page_number"] == 12
    assert result["cached"] is True

    # Upstash's native query: raw text payload, one result.
    mock_upstash_index.query.assert_awaited_once()
    _, kwargs = mock_upstash_index.query.await_args
    assert kwargs["data"] == "What is the refund policy?"
    assert kwargs["top_k"] == 1


async def test_cache_miss_low_similarity(cache_service, mock_upstash_index):
    """Score under the 0.92 gate → None (and the index was still queried)."""
    mock_upstash_index.query.return_value = [
        _query_hit(0.85, {"response": "stale", "agent_type": "rag_agent"})
    ]

    result = await cache_service.get_cached_response(
        "how does refund processing work", "user_123"
    )

    assert result is None
    mock_upstash_index.query.assert_awaited_once()


async def test_cache_miss_empty_results(cache_service, mock_upstash_index):
    """Upstash returns no matches → a safe None."""

    mock_upstash_index.query.return_value = []

    result = await cache_service.get_cached_response(
        "does sahayak support offline mode", "user_123"
    )

    assert result is None
    mock_upstash_index.query.assert_awaited_once()


# ─────────────────────────────────────────────────────────────────────────────
# HARD BYPASS / SAFETY RULES (never cache, never even look up)
# ─────────────────────────────────────────────────────────────────────────────
async def test_bypass_mental_health_query(cache_service, mock_upstash_index):
    """Distress / crisis phrasing short-circuits the cache read entirely."""

    result = await cache_service.get_cached_response(
        "I feel so depressed and need help", "user_123"
    )

    assert result is None
    mock_upstash_index.query.assert_not_awaited()


async def test_bypass_counseling_agent_routing(cache_service, mock_upstash_index):
    """A `counseling_agent` routed response is NEVER persisted."""

    payload = {
        "response": "I am here to support you. Please consider professional help.",
        "routed_agent": "counseling_agent",
        "citations": [],
    }
    await cache_service.set_cached_response("I feel so alone", payload)

    mock_upstash_index.upsert.assert_not_awaited()


async def test_bypass_ephemeral_data(cache_service, mock_upstash_index):
    """Session-scoped / personal queries bypass the cache lookup completely."""

    result = await cache_service.get_cached_response(
        "Summarize my uploaded document", "user_123"
    )

    assert result is None
    mock_upstash_index.query.assert_not_awaited()


# ─────────────────────────────────────────────────────────────────────────────
# WRITE PATH — payload contract under the upsert
# ─────────────────────────────────────────────────────────────────────────────
async def test_cache_set_upsert_payload(cache_service, mock_upstash_index):
    """`set_cached_response` upserts a single vector: id = cache_*, data = raw
    query (Upstash embeds server-side), metadata carries response + agent +
    serialized citations + integer timestamp."""

    payload = {
        "response": "Machine Learning is a subset of AI.",
        "routed_agent": "rag_agent",
        "citations": [{"page_number": 12, "citation_index": 1}],
    }

    await cache_service.set_cached_response("What is Machine Learning?", payload)

    mock_upstash_index.upsert.assert_awaited_once()

    fn = mock_upstash_index.upsert.await_args
    vectors = fn.kwargs.get("vectors")
    assert vectors and len(vectors) == 1

    vector = vectors[0]
    assert vector["id"].startswith("cache_")
    assert vector["data"] == "What is Machine Learning?"

    meta = vector["metadata"]
    assert meta["response"] == payload["response"]
    assert meta["agent_type"] == "rag_agent"
    assert isinstance(meta["timestamp"], int)

    parsed = json.loads(meta["citations"])
    assert parsed[0]["page_number"] == 12
