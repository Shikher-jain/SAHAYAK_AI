"""SQLAlchemy user model for the Sahayak auth system."""
from __future__ import annotations

import datetime

from sqlalchemy import Column, Integer, String, DateTime, Boolean

from backend.auth_system.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    email = Column(String(128), unique=True, index=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(20), nullable=False, default="student")  # student | teacher | admin
    full_name = Column(String(128), default="")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username} role={self.role}>"
