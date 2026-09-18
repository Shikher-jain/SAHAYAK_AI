"""Offline checks for the v2 supervisor/payments logic.
Run inside the v2 runtime venv (see sahayak_ai_v3/requirements.txt); no network needed.
"""
import pathlib

from langgraph.graph import END

from backend.services.agents.supervisor import (
    COUNSELING_EMERGENCY,
    DISTRESS_THRESHOLD,
    parse_distress,
    route_next,
)
from backend.services.payments.user_store import UserStore


def test_parse_distress():
    assert parse_distress("0.95") == 0.95
    assert parse_distress("not-a-number") == 0.0
    assert parse_distress(None) == 0.0
    assert parse_distress("") == 0.0


def test_distress_threshold_matches_docs():
    assert DISTRESS_THRESHOLD == 0.80


def test_route_next_emergency_short_circuits():
    assert route_next({"emergency_flag": True, "next_agent": "counseling_agent"}) == END


def test_route_next_normal():
    assert route_next({"emergency_flag": False, "next_agent": "rag_agent"}) == "rag_agent"
    assert route_next({"emergency_flag": False, "next_agent": "recommender_agent"}) == "recommender_agent"
    assert route_next({"emergency_flag": False, "next_agent": "counseling_agent"}) == "counseling_agent"
    assert route_next({"emergency_flag": False, "next_agent": "unknown"}) == END


def test_emergency_message_carries_helplines():
    assert "14416" in COUNSELING_EMERGENCY          # Tele-MANAS
    assert "9999 666 555" in COUNSELING_EMERGENCY   # Vandrevala


def test_user_store_tier_roundtrip(tmp_path):
    store = UserStore(pathlib.Path(tmp_path) / "p.db")
    assert store.get_tier("u1") == "free"
    assert not store.is_premium("u1")
    store.set_tier("u1", "premium")
    assert store.is_premium("u1")
    assert store.get_tier("u1") == "premium"


def test_user_store_history(tmp_path):
    store = UserStore(pathlib.Path(tmp_path) / "p.db")
    store.append_message("u1", "user", "hello")
    store.append_message("u1", "assistant", "hi there")
    history = store.get_history("u1")
    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_premium_gate(monkeypatch, tmp_path):
    import backend.api.dependencies as deps_mod

    from fastapi import HTTPException

    from backend.api.dependencies import verify_premium_tier

    # patch the user_store reference that verify_premium_tier actually reads
    monkeypatch.setattr(
        deps_mod, "user_store", UserStore(pathlib.Path(tmp_path) / "g.db")
    )

    try:
        verify_premium_tier(user_id="u2")
        raise AssertionError("free user should have been rejected")
    except HTTPException as e:
        assert e.status_code == 403

    deps_mod.user_store.set_tier("u2", "premium")
    assert verify_premium_tier(user_id="u2") == "u2"