"""SQLAlchemy model for persisted conversation history.

Uses the same Base/engine as the auth system (backend/auth_system/database.py)
— one Postgres (Neon) connection, one set of tables, no separate DB setup.
Table is created automatically via init_db() at startup, same as `users`.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text, Index

from backend.auth_system.database import Base


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False)
    role = Column(String(16), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        # Every query filters by session_id and orders by created_at —
        # this composite index keeps get_history() fast even as the table
        # grows across many sessions.
        Index("ix_conversation_session_created", "session_id", "created_at"),
    )