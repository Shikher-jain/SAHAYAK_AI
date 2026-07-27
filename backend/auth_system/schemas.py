"""Pydantic schemas for request/response validation in the auth system."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# --- Request schemas ---

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    email: str = Field(..., max_length=128)
    password: str = Field(..., min_length=6, max_length=128)
    full_name: str = Field(default="", max_length=128)
    # NOTE: role intentionally NOT accepted here. All new accounts are
    # "student" by default (enforced server-side in register_user()).
    # Promoting a user to teacher/admin is an admin-only action —
    # see PATCH /auth/users/{user_id}/role.


class UserLogin(BaseModel):
    username: str
    password: str


class UserUpdate(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    # NOTE: role intentionally NOT accepted here — see UserCreate note above.


class AdminRoleUpdate(BaseModel):
    role: str = Field(..., pattern="^(student|teacher|admin)$")


# --- Response schemas ---

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None


class UserProfile(BaseModel):
    id: int
    username: str
    email: str
    role: str
    full_name: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True