"""SQLite database setup and session factory for the auth system."""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Database path — stored alongside local stack data for consistency.
_BASE_DIR = Path(__file__).resolve().parents[2]
_DB_DIR = _BASE_DIR / "data" / "auth"
_DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = _DB_DIR / "sahayak_auth.db"

DATABASE_URL = os.getenv("AUTH_DATABASE_URL", f"sqlite:///{DB_PATH}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite only
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Session:
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they do not exist."""
    from backend.auth_system import models  # noqa: F401  ensure models are registered
    Base.metadata.create_all(bind=engine)
