"""Postgres-backed conversational history keyed by session_id.

Was in-memory (a plain dict) — meant chat history was lost on every server
restart/redeploy, including Render's free-tier spin-down after inactivity.
Now persisted in the same Postgres (Neon) database as auth, via
ConversationMessage (see conversation_models.py).

Same public interface as before (add_exchange/get_history/clear) so
backend/services/vector_service.py doesn't need any changes.
"""
from __future__ import annotations

import logging

from backend.auth_system.database import SessionLocal
from backend.rag.conversation_models import ConversationMessage

logger = logging.getLogger(__name__)


class ConversationManager:
    """Per-session sliding window of user/assistant turns, backed by Postgres."""

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size

    def add_exchange(self, session_id: str, query: str, answer: str) -> None:
        db = SessionLocal()
        try:
            db.add(ConversationMessage(session_id=session_id, role="user", content=query))
            db.add(ConversationMessage(session_id=session_id, role="assistant", content=answer))
            db.commit()
            self._trim_old_messages(db, session_id)
        except Exception:
            logger.exception("Failed to persist conversation exchange for session %s", session_id)
            db.rollback()
        finally:
            db.close()

    def get_history(self, session_id: str) -> str:
        db = SessionLocal()
        try:
            rows = (
                db.query(ConversationMessage)
                .filter(ConversationMessage.session_id == session_id)
                .order_by(ConversationMessage.created_at.desc())
                .limit(self.window_size * 2)
                .all()
            )
            rows.reverse()
            formatted = [f"{'User' if r.role == 'user' else 'Assistant'}: {r.content}" for r in rows]
            return "\n".join(formatted).strip()
        except Exception:
            logger.exception("Failed to load conversation history for session %s", session_id)
            return ""
        finally:
            db.close()

    def clear(self, session_id: str) -> None:
        db = SessionLocal()
        try:
            db.query(ConversationMessage).filter(ConversationMessage.session_id == session_id).delete()
            db.commit()
        except Exception:
            logger.exception("Failed to clear conversation history for session %s", session_id)
            db.rollback()
        finally:
            db.close()

    def _trim_old_messages(self, db, session_id: str) -> None:
        """Keep only the most recent window_size*2 messages per session —
        prevents the table from growing unbounded for long-running sessions
        (unlike the old in-memory deque, a DB table has no automatic size cap)."""
        keep_limit = self.window_size * 2
        ids_to_keep = {
            row.id
            for row in (
                db.query(ConversationMessage.id)
                .filter(ConversationMessage.session_id == session_id)
                .order_by(ConversationMessage.created_at.desc())
                .limit(keep_limit)
                .all()
            )
        }
        if not ids_to_keep:
            return
        (
            db.query(ConversationMessage)
            .filter(
                ConversationMessage.session_id == session_id,
                ~ConversationMessage.id.in_(ids_to_keep),
            )
            .delete(synchronize_session=False)
        )
        db.commit()